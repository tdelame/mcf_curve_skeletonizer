/* Created on: Nov 18, 2015
 *     Author: T.Delame (tdelame@gmail.com)
 */
# include <skeletonizer.h>

# include <list>
# include <omp.h>
# include <flann/flann.hpp>
# include <Eigen/Core>
# include <Eigen/Sparse>
BEGIN_PROJECT_NAMESPACE

  skeletonizer::parameters::parameters()
    : omega_velocity{ 0.1 }, omega_medial{ 0.4 },
      edge_length_threshold{ -1 },
      zero_threshold{ 1e-6 }, inv_zero_threshold{ 1e6 }
  {}


  skeletonizer::skeletonizer(
      triangular_mesh& mesh,
      const parameters& params )
    : mesh{ mesh }, params{ params }, duration{ 0 }
  {
    this->params.inv_zero_threshold = real(1.0)/params.zero_threshold;
  }

  void
  skeletonizer::iterate()
  {
    auto start = omp_get_wtime();
    if( !cotangent_weight_handle.is_valid() )
      initialize();
    contract_geometry();
    update_topology();
    detect_degeneracies();
    duration += omp_get_wtime() - start;
  }

  void
  skeletonizer::convert_to_skeleton()
  {
    LOG( warning, "todo");
    mesh.garbage_collection( true, true, true );
  }

  real
  skeletonizer::get_execution_duration() const noexcept
  {
    return duration;
  }

  enum {
    STATUS_OK = 0, STATUS_FIXED = 1, STATUS_COLLAPSED = 2, STATUS_SPLITTED = 4
  };

  static real
  compute_radius(
    const triangular_mesh::Point& sample,
    const triangular_mesh::Point& point,
    const triangular_mesh::Point& normal )
  {
    auto diff = sample - point;
    return dot(diff, diff) / (real(2.0) * std::abs(dot(normal, diff)));
  }

  static const real min_radius_variation = 1e-6;

  void
  skeletonizer::initialize()
  {
    // some deleted elements can already exist in the mesh
    mesh.garbage_collection( true, true, true );

    const auto nsamples = mesh.n_vertices();

    /*** resize and initialize internal data ***/
    mesh.add_property( cotangent_weight_handle ); // recomputed before using it
    mesh.add_property( alpha_handle );            // recomputed before using it
    mesh.add_property( omega_L_handle );          // need to be initialized here
    mesh.add_property( omega_V_handle );          // need to be initialized here
    mesh.add_property( omega_M_handle );          // need to be initialized here
    mesh.add_property( medial_handle );           // need to be initialized here
    mesh.add_property( status_handle );           // need to be initialized here

    /**
     * Note: we need the medial point cloud to compute the medial energy.
     * In the Mean Curve Skeletonization, a powershape algorithm was used.
     * It requires a 3D Voronoi Diagram construction. It seems a little overkill
     * just to compute the medial point cloud (we ignore the medial surface
     * topology). This is why I propose instead to use the so-called shrinking
     * ball algorithm, quite simple and can be run in parallel. The method was
     * originaly described in the following paper:
     *
     * Jaehwan Ma, Sang Won Bae, and Sunghee Choi.
     * 3D medial axis point approximation using nearest neighbors and the normal
     * field. The Visual Computer, 28(1):7–19, 2012
     *
     * This part can be replaced by any other algorithm you want.
     */

    /*** Acquire positions, normals and bounding box***/
    auto shape_samples = flann::Matrix<real> { new real[nsamples * 3], nsamples, 3 };
    auto point_max = point{ -REAL_MAX, -REAL_MAX, -REAL_MAX };
    auto point_min = point{ REAL_MAX, REAL_MAX, REAL_MAX };
    # pragma omp parallel
    {
      auto thread_max = point{ -REAL_MAX, -REAL_MAX, -REAL_MAX };
      auto thread_min = point{ REAL_MAX, REAL_MAX, REAL_MAX };
      # pragma omp for
      for( uint32_t i = 0; i < nsamples; ++i )
       {
         auto h = vhandle( i );
         laplacian_weight( h ) = 1.0;
         velocity_weight( h ) = params.omega_velocity;
         medial_weight( h ) = params.omega_medial;
         status( h ) = STATUS_OK;

         auto& source = mesh.point( h );
         for( int c = 0; c < 3; ++ c )
           {
             shape_samples[i][c] = source[c];
             if( source[c] < thread_min[c] )
               thread_min[c] = source[c];
             if( source[c] > thread_max[c] )
               thread_max[c] = source[c];
           }
       }
      # pragma omp critical
      {
        for( int c = 0; c < 3; ++ c )
          {
            point_max[c] = std::max( point_max[c], thread_max[c] );
            point_min[c] = std::min( point_min[c], thread_min[c] );
          }
      }
    }
    point_max -= point_min;

    if( params.edge_length_threshold <= 0 )
      {
        params.edge_length_threshold = 1e-3 * point_max.length();
      }

    /** beware
     * A small global initial radius will greatly speed-up the process.
     * However, when the initial radius is smaller than the radius of the
     * maximal included ball passing (and tangent) to the sample, the produced
     * atom won't be maximal! This initial global radius need to be greater than
     * all atom radii. As we do not have access to radii yet (this is what we
     * are trying to compute), we have to guess.
     */
    auto global_initial_radius = 0.2 * std::min( point_max[0], std::min( point_max[1], point_max[2] ) );

    /*** set up a kdtree to index the positions ***/
    auto shape_samples_index = flann::Index<flann::L2_3D<real> > {
       shape_samples, flann::KDTreeSingleIndexParams { 32 } };
    shape_samples_index.buildIndex();
    flann::SearchParams knn_search( -1, 0.0, true );

    /*** shrinking ball algorithm in parallel ***/
    # pragma omp parallel
    {
     // variables for the knn search with the FLANN library
     auto flann_query = flann::Matrix<real>( new real[3], 1, 3 );
     auto flann_indices = flann::Matrix<size_t>( new size_t[2], 1, 2 );
     auto flann_distances = flann::Matrix<real>( new real[2], 1, 2 );

     # pragma omp for schedule(dynamic)
     for (uint32_t sampleid = 0; sampleid < nsamples; ++sampleid)
       {
         // data for this vertex (reminder: one atom by sample)
         const auto handle = vhandle( sampleid );
         const auto& sample_point = mesh.point( handle );
         const auto& sample_normal = mesh.normal( handle );

         // initialization: a big ball tangent to this sample
         auto radius = global_initial_radius;
         auto center = sample_point - radius * sample_normal;

         // search the closest sample to the center that is different from current sample
         flann_query[0][0] = center[0];
         flann_query[0][1] = center[1];
         flann_query[0][2] = center[2];
         shape_samples_index.knnSearch( flann_query, flann_indices, flann_distances, 2, knn_search );

         // compute the radius of the ball tangent to the current sample and touching the other sample
         auto other_index = (flann_indices[0][0] == sampleid) ? flann_indices[0][1] : flann_indices[0][0];
         auto other_sample = mesh.point( vhandle( other_index ) );
         auto next_radius = compute_radius( sample_point, other_sample, sample_normal );

         while (std::abs( next_radius - radius ) > min_radius_variation)
             {
               radius = next_radius;
               center = sample_point - radius * sample_normal;

               flann_query[0][0] = center[0];
               flann_query[0][1] = center[1];
               flann_query[0][2] = center[2];
               shape_samples_index.knnSearch( flann_query, flann_indices, flann_distances, 2, knn_search );

               other_index = (flann_indices[0][0] == sampleid) ? flann_indices[0][1] : flann_indices[0][0];
               other_sample = mesh.point( vhandle( other_index ) );
               next_radius = compute_radius( sample_point, other_sample, sample_normal );
             }
           if( !std::isfinite( center[0] ) || !std::isfinite( center[1] ) || !std::isfinite( center[2] ) )
             {
               LOG(error, "got pole = " << center << " for atom #" << sampleid << " and radius = " << radius );
             }
           medial_point( handle ) = center;
         }

       // free the resources used for knn search
       delete[] flann_query.ptr();
       delete[] flann_indices.ptr();
       delete[] flann_distances.ptr();
     }
    delete[] shape_samples.ptr();
  }

  void
  skeletonizer::fix( const vhandle& handle )
  {
    status( handle ) |= STATUS_FIXED;
    laplacian_weight( handle ) = real(0);
    velocity_weight( handle ) = params.inv_zero_threshold;
    medial_weight( handle ) = real(0);
  }

  static real
  clamped_cotangent_edge_weight(
    const triangular_mesh::Point& a,
    const triangular_mesh::Point& b )
  {
    static const real lower_bound = -0.999;
    static const real upper_bound = 0.999;
    return 1.0 / std::tan( std::acos( std::min( upper_bound, std::max( dot( a, b ), lower_bound ) ) ) );
  }

  void
  skeletonizer::contract_geometry()
  {
    const auto all_edges = mesh.n_edges();
    const auto all_vertices = mesh.n_vertices();

    /**************************************************************************
     * Edge data computation:                                                 *
     *   - compute cotangent weights                                          *
     **************************************************************************/
    # pragma omp parallel for
    for( uint64_t i = 0; i < all_edges; ++ i )
      {
        ehandle h( i );
        real weight = 0;
        hehandle h0( i << 1 );
        hehandle h1( h0.idx() + 1 );

        if( mesh.is_boundary( h0 ) || mesh.is_boundary( h1 ) )
          throw std::logic_error("non manifold edge should not happen at this stage");

        const auto& p0 = mesh.point( mesh.to_vertex_handle( h0 ) );
        const auto& p1 = mesh.point( mesh.to_vertex_handle( h1 ) );
        const auto& p2 = mesh.point( mesh.to_vertex_handle( mesh.next_halfedge_handle( h0 ) ) );
        const auto& p3 = mesh.point( mesh.to_vertex_handle( mesh.next_halfedge_handle( h1 ) ) );

        weight = std::max( real(0.0),
          clamped_cotangent_edge_weight( (p0 - p2).normalized(), (p1 - p2).normalized() )
        + clamped_cotangent_edge_weight( (p0 - p3).normalized(), (p1 - p3).normalized() ));
        cotangent_weight( h ) = weight;
      }

    /*** declare and reserve enough space for Eigen structures***/
    const auto ncols = all_vertices;
    const auto nrows = 3 * ncols;
    Eigen::SparseMatrix<real> LHS( nrows, ncols );
    LHS.reserve( all_edges * 2 + nrows );
    Eigen::MatrixXd RHS = Eigen::MatrixXd::Zero( nrows, 3 );
    std::vector<Eigen::Triplet<real> > triplets; //will be used to fill LHS
    triplets.resize( (all_edges<<1) + nrows, Eigen::Triplet<real>( 0, 0, 0 ) );

    /*** fill RHS and LHS ***/
    # pragma omp parallel for
    for( uint64_t i = 0; i < all_edges; ++ i )
      {
        ehandle h( i );
        const auto index = i << 1;
        hehandle h0( index );
        hehandle h1( index + 1 );
        auto i0 = mesh.to_vertex_handle( h0 );
        auto i1 = mesh.to_vertex_handle( h1 );
        triplets[ i << 1 ] = Eigen::Triplet< real >( i0.idx(), i1.idx(),
            cotangent_weight( h ) * laplacian_weight( i0 ) );
        triplets[ ( i << 1) + 1 ] = Eigen::Triplet< real >(
            i1.idx(), i0.idx(),
            cotangent_weight( h ) * laplacian_weight( i1 ) );
      }

    # pragma omp parallel for schedule(dynamic)
    for( uint32_t i = 0; i < all_vertices; ++ i )
      {
        vhandle h( i );
        auto tid = (all_edges << 1) + 3 * i;
        real sum = 0;
        for( auto it = mesh.ve_begin( h ), end = mesh.ve_end( h ); it != end; ++ it )
          sum += cotangent_weight( *it );
        triplets[ tid ] = Eigen::Triplet<real>( i, i, -sum );
        triplets[ tid + 1 ] = Eigen::Triplet<real>( i + ncols, i, velocity_weight( h ) );
        triplets[ tid + 2 ] = Eigen::Triplet<real>( i + 2 * ncols, i, medial_weight( h ) );
        auto u = mesh.point( h ) * velocity_weight( h );
        RHS.row( ncols + i ) = Eigen::Vector3d( u[0], u[1], u[2] );
        u = medial_point( h ) * medial_weight( h );
        RHS.row( (ncols << 1) + i ) = Eigen::Vector3d( u[0], u[1], u[2] );
      }
    LHS.setFromTriplets( triplets.begin(), triplets.end() );

    /*** solve the linear system ***/
    {
      Eigen::SparseMatrix<real> At = LHS.transpose();
      Eigen::SparseMatrix<real> AtA = At * LHS;
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<real> > solver;
      solver.compute( AtA );
      Eigen::MatrixXd X = Eigen::MatrixXd::Zero( ncols, 3 );
      X.col( 0 ) = solver.solve( At * RHS.col( 0 ) );
      X.col( 1 ) = solver.solve( At * RHS.col( 1 ) );
      X.col( 2 ) = solver.solve( At * RHS.col( 2 ) );
      if( !std::isfinite( X.norm() ) )
        {
          throw std::runtime_error(
            "something went wrong in the linear least square solution X(0,0)" );
        }
      # pragma omp parallel for schedule(dynamic)
      for( uint32_t i = 0; i < all_vertices; ++ i )
        {
          vhandle h( i );
          auto p = X.row( i );
          mesh.set_point( h, point{ p[0], p[1], p[2] } );
        }
    }
  }

  void
  skeletonizer::detect_degeneracies()
  {
    //todo: understand why in the original code there was the numeric constant 0.01 (0.1 * 0.1)
    const auto sqlength = params.edge_length_threshold * params.edge_length_threshold * 0.01;
    const auto nsamples = mesh.n_vertices();

    # pragma omp parallel for schedule(dynamic)
    for( uint32_t i = 0; i < nsamples; ++ i )
      {
        vhandle h( i );
        if( !(status(h) & STATUS_FIXED ) )
          {
            const auto& p = mesh.point( h );
            char ok = 2; //if number of bad edges >= 2, this vertex is fixed
            for( auto it = mesh.voh_begin( h ), end = mesh.voh_end( h ); it != end; ++ it )
              {
                if( ( p - mesh.point( mesh.to_vertex_handle( *it ) ) ).sqrnorm() < sqlength
                    && !mesh.is_collapse_ok( *it ) )
                  {
                    --ok;
                    if( !ok )
                      {
                        fix( h );
                        break;
                      }
                  }
              }
          }
      }
  }

  static bool
  compare_edge_length(
      const std::pair< triangular_mesh::EdgeHandle, real >& a,
      const std::pair< triangular_mesh::EdgeHandle, real >& b )
  {
    return a.second < b.second;
  }

  static bool
  compare_edge(
      const std::pair< triangular_mesh::EdgeHandle, real >& a,
      const std::pair< triangular_mesh::EdgeHandle, real >& b )
  {
    return ( a.first == b.first ) ? a.second < b.second : a.first.idx() < b.first.idx();
  }

  static bool
  are_same_edge(
      const std::pair< triangular_mesh::EdgeHandle, real >& a,
      const std::pair< triangular_mesh::EdgeHandle, real >& b )
  {
    return a.first == b.first;
  }

  void
  skeletonizer::update_topology()
  {
    //FIXME: so many things to improve here...

      //collapse too short links
      {
        const auto sqelength = params.edge_length_threshold * params.edge_length_threshold;
        const auto nedges = mesh.n_edges();
        // obtain quickly the first set of links to collapse
        std::list< std::pair<ehandle, real> > tocollapse;
        # pragma omp parallel for schedule(dynamic)
        for( uint64_t i = 0; i < nedges; ++ i )
          {
            ehandle h( i );
            hehandle h0( i << 1 );
            hehandle h1( h0.idx() + 1 );
            vhandle v0( mesh.to_vertex_handle( h0 ) );
            const auto& p0 = mesh.point( v0 );
            const auto& p1 = mesh.point( mesh.to_vertex_handle( h1 ) );
            const auto l2 = (p0 - p1).sqrnorm();
            if( l2 < sqelength && mesh.is_collapse_ok( h0 ) )
              {
                # pragma omp critical
                tocollapse.push_back( { h, l2 } );
              }
          }
        tocollapse.sort( compare_edge_length );

        while( !tocollapse.empty() )
          {
            std::list< std::pair<ehandle, real> > next;
            while( !tocollapse.empty() )
              {
                if( !mesh.status( tocollapse.front().first ).deleted() )
                  {
                    hehandle h0( tocollapse.front().first.idx() << 1 );
                    hehandle h1( h0.idx() + 1 );
                    vhandle v0( mesh.to_vertex_handle( h0 ) );
                    vhandle v1( mesh.to_vertex_handle( h1 ) );
                    auto& p0 = mesh.point( v0 );
                    const auto& p1 = mesh.point( v1 );
                    if( (p0 - p1).sqrnorm() < sqelength && mesh.is_collapse_ok( h0 ) )
                      {
                        // vertex v0 will remain after the collapse operation
                        p0 = 0.5 * ( p0 + p1 );
                        bool fixed0 = status( v0 ) & STATUS_FIXED;
                        bool fixed1 = status( v1 ) & STATUS_FIXED;
                        if( (medial_point( v0 ) - p0).sqrnorm() > (medial_point(v1) - p0).sqrnorm() )
                          medial_point(v0) = medial_point(v1);
                        mesh.collapse( h0 );
                        if( fixed0 || fixed1 ) fix( v0 );

                        for( auto it = mesh.ve_begin( v0 ), end = mesh.ve_end( v0 ); it != end; ++ it )
                          {
                            auto he = hehandle( it->idx() << 1 );
                            auto v2 = mesh.to_vertex_handle( he ) == v0 ? mesh.from_vertex_handle( he ) : mesh.to_vertex_handle( he );
                            auto l2 = ( mesh.point( v2 ) - p0 ).sqrnorm();
                            if( l2 < sqelength )
                              {
                                next.push_back( { *it, l2 } );
                              }
                          }
                      }
                  }
                tocollapse.pop_front();
              }
            next.sort( compare_edge );
            next.unique( are_same_edge );
            tocollapse.swap( next );
          }
      }

      //split too flat triangles
      {
        //todo: implement a more efficient algorithm
        bool modifications = true;
        const static real alpha = 110.0 * M_PI / 180.0;
        while( modifications )
          {
            const auto nfaces = mesh.n_faces();
            # pragma omp parallel for schedule(dynamic)
            for( uint64_t i = 0; i < nfaces; ++ i )
              {
                fhandle h( i );
                if( !mesh.status( h ).deleted() )
                  {
                    auto h_a = mesh.halfedge_handle( h );
                    auto h_b = mesh.next_halfedge_handle( h_a );
                    auto h_c = mesh.next_halfedge_handle( h_b );

                    real a = (mesh.point( mesh.to_vertex_handle( h_a ) ) - mesh.point( mesh.from_vertex_handle( h_a ) ) ).norm(), a2 = a * a;
                    real b = (mesh.point( mesh.to_vertex_handle( h_b ) ) - mesh.point( mesh.from_vertex_handle( h_b ) ) ).norm(), b2 = b * b;
                    real c = (mesh.point( mesh.to_vertex_handle( h_c ) ) - mesh.point( mesh.from_vertex_handle( h_c ) ) ).norm(), c2 = c * c;

                    /// A degenerate triangle will never undergo a split (but rather a collapse...)
                    if( a< params.edge_length_threshold || b< params.edge_length_threshold  || c< params.edge_length_threshold )
                      {
                        halpha( h_a ) = -1;
                        halpha( h_b ) = -1;
                        halpha( h_c ) = -1;
                      }
                    else
                      {
                        /// Opposite angles (from law of cosines)
                        halpha( h_a ) = std::acos( std::max( -1.0, std::min( 1.0, (-a2 +b2 +c2)/(2*  b*c) ) ) );
                        halpha( h_b ) = std::acos( std::max( -1.0, std::min( 1.0, (+a2 -b2 +c2)/(2*a  *c) ) ) );
                        halpha( h_c ) = std::acos( std::max( -1.0, std::min( 1.0, (+a2 +b2 -c2)/(2*a*b  ) ) ) );
                      }
                  }
              }

            modifications = false;
            const auto nedges = mesh.n_edges();
            for( uint64_t i = 0; i < nedges; ++ i )
              {
                ehandle h( i );
                if( !mesh.status( h ).deleted() )
                  {
                    hehandle h0( i << 1 );
                    hehandle h1( h0.idx() + 1 );
                    vhandle v0 = mesh.to_vertex_handle( h0 );
                    vhandle v1 = mesh.to_vertex_handle( h1 );

                    /// Should a split take place?
                    real alpha_0 = halpha( h0 );
                    real alpha_1 = halpha( h1 );
                    if( alpha_0 < alpha || alpha_1 < alpha ) continue;

                    /// Which side should I split?
                    auto w0 = mesh.to_vertex_handle( mesh.next_halfedge_handle( h0 ) );
                    auto w1 = mesh.to_vertex_handle( mesh.next_halfedge_handle( h1 ) );
                    auto wsplitside = (alpha_0>alpha_1) ? w0 : w1;

                    /// Project side vertex on edge
                    point p0 = mesh.point( v0 );
                    auto projector = (mesh.point( v1 )-p0).normalized();
                    auto t = dot(projector, mesh.point( wsplitside ) - p0);
//
//                    auto new_pole = medial_point( v0 ) + t * ( medial_point(v1) - medial_point(v0) ).normalize();
                    auto vnew = mesh.split( h, p0 + t * projector );

                    /// Also project the pole
//                    medial_point( vnew ) = new_pole;
                    laplacian_weight( vnew ) = real(1.0);
                    velocity_weight( vnew ) = params.omega_velocity;
                    medial_weight( vnew ) = 0;// params.omega_medial;

                    modifications = true;
                    /// And mark it as a split
                    //fixme: why processing splitted vertices differently? Indeed, if we compute another medial point,
                    // why ignoring it by setting a null medial_weight in the original code?
  //                  status( vnew ) |= STATUS_SPLITTED;
                  }
              }
          }
      }

      mesh.garbage_collection( true, true, true );
    }
END_PROJECT_NAMESPACE
