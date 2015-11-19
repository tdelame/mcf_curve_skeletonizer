/*  Created on: Feb 22, 2015
 *      Author: T.Delame (tdelame@gmail.com)
 */
# include <OpenMesh/Core/IO/exporter/BaseExporter.hh>
# include <OpenMesh/Core/IO/IOManager.hh>

# include <mesh.h>
# include <fstream>
# include <iomanip>

BEGIN_PROJECT_NAMESPACE

  static unsigned char
  cc_to_uc( const double& value )
  {
    return (uint)(value*255.0);
  }

  static unsigned int
  cc_to_ui( const double& value )
  {
    return (uint)(value*65535);
  }

  class ExporterT : public OpenMesh::IO::BaseExporter {
  public:
    virtual ~ExporterT(){}

    // Constructor
    ExporterT(const triangular_mesh& _mesh) : mesh_(_mesh) {}


    // get vertex data

    OpenMesh::Vec3f  point(triangular_mesh::VertexHandle _vh)    const
    {
      auto point = mesh_.point(_vh);
      return OpenMesh::Vec3f( point[0], point[1], point[2] );
    }

    OpenMesh::Vec3f  normal(triangular_mesh::VertexHandle _vh)   const
    {
      auto p = mesh_.normal(_vh);
      return OpenMesh::Vec3f( p[0], p[1], p[2] );
    }

    OpenMesh::Vec3uc color(triangular_mesh::VertexHandle _vh)    const
    {
      auto c = mesh_.color(_vh);
      return OpenMesh::Vec3uc( cc_to_uc(c[0]), cc_to_uc(c[1]), cc_to_uc(c[2]));
    }

    OpenMesh::Vec4uc colorA(triangular_mesh::VertexHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4uc( cc_to_uc(c[0]), cc_to_uc(c[1]), cc_to_uc(c[2]), cc_to_uc(c[3]));;
    }

    OpenMesh::Vec3ui colori(triangular_mesh::VertexHandle h)    const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3ui( cc_to_ui(c[0]), cc_to_ui(c[1]), cc_to_ui(c[2]));
    }

    OpenMesh::Vec4ui colorAi(triangular_mesh::VertexHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4ui( cc_to_ui(c[0]), cc_to_ui(c[1]), cc_to_ui(c[2]), cc_to_ui(c[3]));;
    }

    OpenMesh::Vec3f colorf(triangular_mesh::VertexHandle h)    const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3f( c[0], c[1], c[2] );
    }

    OpenMesh::Vec4f colorAf(triangular_mesh::VertexHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4f( c[0], c[1], c[2], c[3] );
    }

    OpenMesh::Vec2f  texcoord(triangular_mesh::VertexHandle _vh) const
    {
      return mesh_.texcoord2D(_vh);
    }

    // get edge data

    OpenMesh::Vec3uc color(triangular_mesh::EdgeHandle h)    const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3uc( cc_to_uc(c[0]), cc_to_uc(c[1]), cc_to_uc(c[2]));
    }

    OpenMesh::Vec4uc colorA(triangular_mesh::EdgeHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4uc( cc_to_uc(c[0]), cc_to_uc(c[1]), cc_to_uc(c[2]), cc_to_uc(c[3]));;
    }

    OpenMesh::Vec3ui colori(triangular_mesh::EdgeHandle h)    const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3ui( cc_to_ui(c[0]), cc_to_ui(c[1]), cc_to_ui(c[2]));
    }

    OpenMesh::Vec4ui colorAi(triangular_mesh::EdgeHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4ui( cc_to_ui(c[0]), cc_to_ui(c[1]), cc_to_ui(c[2]), cc_to_ui(c[3]));;
    }

    OpenMesh::Vec3f colorf(triangular_mesh::EdgeHandle h)    const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3f( c[0], c[1], c[2] );
    }

    OpenMesh::Vec4f colorAf(triangular_mesh::EdgeHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4f( c[0], c[1], c[2], c[3] );;
    }

    // get face data

    unsigned int get_vhandles(triangular_mesh::FaceHandle _fh,
            std::vector<triangular_mesh::VertexHandle>& _vhandles) const
    {
      unsigned int count(0);
      _vhandles.clear();
      for (typename triangular_mesh::CFVIter fv_it=mesh_.cfv_iter(_fh); fv_it.is_valid(); ++fv_it)
      {
        _vhandles.push_back(*fv_it);
        ++count;
      }
      return count;
    }

    OpenMesh::Vec3f  normal(triangular_mesh::FaceHandle h)   const
    {
      auto n = mesh_.normal(h);
      return OpenMesh::Vec3f( n[0], n[1], n[2] );
    }

    OpenMesh::Vec3uc  color(triangular_mesh::FaceHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3uc( cc_to_uc(c[0]), cc_to_uc(c[1]), cc_to_uc(c[2]));
    }

    OpenMesh::Vec4uc  colorA(triangular_mesh::FaceHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4uc( cc_to_uc(c[0]), cc_to_uc(c[1]), cc_to_uc(c[2]), cc_to_uc(c[3]));;
    }

    OpenMesh::Vec3ui  colori(triangular_mesh::FaceHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3ui( cc_to_ui(c[0]), cc_to_ui(c[1]), cc_to_ui(c[2]));
    }

    OpenMesh::Vec4ui  colorAi(triangular_mesh::FaceHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4ui( cc_to_ui(c[0]), cc_to_ui(c[1]), cc_to_ui(c[2]), cc_to_ui(c[3]));;
    }

    OpenMesh::Vec3f colorf(triangular_mesh::FaceHandle h)    const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec3f( c[0], c[1], c[2] );
    }

    OpenMesh::Vec4f colorAf(triangular_mesh::FaceHandle h)   const
    {
      auto c = mesh_.color(h);
      return OpenMesh::Vec4f( c[0], c[1], c[2], c[3] );;
    }

    virtual const OpenMesh::BaseKernel* kernel() { return &mesh_; }


    // query number of faces, vertices, normals, texcoords
    size_t n_vertices()  const { return mesh_.n_vertices(); }
    size_t n_faces()     const { return mesh_.n_faces(); }
    size_t n_edges()     const { return mesh_.n_edges(); }


    // property information
    bool is_triangle_mesh() const
    { return triangular_mesh::is_triangles(); }

    bool has_vertex_normals()   const { return mesh_.has_vertex_normals();   }
    bool has_vertex_colors()    const { return mesh_.has_vertex_colors();    }
    bool has_vertex_texcoords() const { return mesh_.has_vertex_texcoords2D(); }
    bool has_edge_colors()      const { return mesh_.has_edge_colors();      }
    bool has_face_normals()     const { return mesh_.has_face_normals();     }
    bool has_face_colors()      const { return mesh_.has_face_colors();      }

  private:

     const triangular_mesh& mesh_;
  };


  static const real inv_255 = real( 1.0 /  255.0 );

  static real
  cc_from_uc( const unsigned char& value )
  {
    return real( value ) * inv_255;
  }

  class ImporterT : public OpenMesh::IO::BaseImporter
  {
  public:

    typedef typename triangular_mesh::Point       Point;
    typedef typename triangular_mesh::Normal      Normal;
    typedef typename triangular_mesh::Color       Color;
    typedef typename triangular_mesh::TexCoord2D  TexCoord2D;
    typedef typename triangular_mesh::VertexHandle VertexHandle;
    typedef typename triangular_mesh::EdgeHandle EdgeHandle;
    typedef typename triangular_mesh::HalfedgeHandle HalfedgeHandle;
    typedef typename triangular_mesh::FaceHandle FaceHandle;
    typedef std::vector<VertexHandle>  VHandles;


    ImporterT(triangular_mesh& _mesh) : mesh(_mesh), halfedgeNormals_() {}


    virtual VertexHandle add_vertex(const OpenMesh::Vec3f& p)
    {
      return mesh.add_vertex( Point(p[0], p[1], p[2] ) );
    }

    virtual VertexHandle add_vertex()
    {
      return mesh.new_vertex();
    }

    virtual FaceHandle add_face(const VHandles& _indices)
    {
      FaceHandle fh;

      if (_indices.size() > 2)
      {
        VHandles::const_iterator it, it2, end(_indices.end());


        // test for valid vertex indices
        for (it=_indices.begin(); it!=end; ++it)
          if (! mesh.is_valid_handle(*it))
          {
            omerr() << "ImporterT: Face contains invalid vertex index\n";
            return fh;
          }


        // don't allow double vertices
        for (it=_indices.begin(); it!=end; ++it)
          for (it2=it+1; it2!=end; ++it2)
            if (*it == *it2)
            {
              omerr() << "ImporterT: Face has equal vertices\n";
              failed_faces_.push_back(_indices);
              return fh;
            }


        // try to add face
        fh = mesh.add_face(_indices);
        if (!fh.is_valid())
        {
          failed_faces_.push_back(_indices);
          return fh;
        }

        //write the half edge normals
        if (mesh.has_halfedge_normals())
        {
          //iterate over all incoming haldedges of the added face
          for (typename triangular_mesh::FaceHalfedgeIter fh_iter = mesh.fh_begin(fh);
              fh_iter != mesh.fh_end(fh); ++fh_iter)
          {
            //and write the normals to it
            typename triangular_mesh::HalfedgeHandle heh = *fh_iter;
            typename triangular_mesh::VertexHandle vh = mesh.to_vertex_handle(heh);
            typename std::map<VertexHandle,Normal>::iterator it_heNs = halfedgeNormals_.find(vh);
            if (it_heNs != halfedgeNormals_.end())
              mesh.set_normal(heh,it_heNs->second);
          }
          halfedgeNormals_.clear();
        }
      }
      return fh;
    }

    // vertex attributes

    virtual void set_point(VertexHandle _vh, const OpenMesh::Vec3f& p)
    {
      mesh.set_point(_vh, Point( p[0], p[1], p[3] ) );
    }

    virtual void set_normal(VertexHandle _vh, const OpenMesh::Vec3f& n )
    {
      auto normal = Normal( n[0], n[1], n[3] );
      if (mesh.has_vertex_normals())
        mesh.set_normal(_vh, normal );

      //saves normals for half edges.
      //they will be written, when the face is added
      if (mesh.has_halfedge_normals())
        halfedgeNormals_[_vh] = normal;
    }

    virtual void set_color(VertexHandle _vh, const OpenMesh::Vec4uc& c )
    {
      if (mesh.has_vertex_colors())
        mesh.set_color( _vh, Color( cc_from_uc(c[0]), cc_from_uc(c[1]), cc_from_uc(c[2]), cc_from_uc(c[3]) ) );
    }

    virtual void set_color(VertexHandle _vh, const OpenMesh::Vec3uc& c)
    {
      if (mesh.has_vertex_colors())
        mesh.set_color( _vh, Color( cc_from_uc(c[0]), cc_from_uc(c[1]), cc_from_uc(c[2]), 1.0 ) );
    }

    virtual void set_color(VertexHandle h, const OpenMesh::Vec4f& c )
    {
      if (mesh.has_vertex_colors())
        mesh.set_color( h, Color( c[0], c[1], c[2], c[3] ) );
    }

    virtual void set_color(VertexHandle _vh, const OpenMesh::Vec3f& c)
    {
      if (mesh.has_vertex_colors())
        mesh.set_color( _vh, Color( c[0], c[1], c[2], 1.0 ) );
    }

    virtual void set_texcoord(VertexHandle _vh, const OpenMesh::Vec2f& _texcoord)
    {
      if (mesh.has_vertex_texcoords2D())
        mesh.set_texcoord2D(_vh, _texcoord );
    }

    virtual void set_texcoord(HalfedgeHandle _heh, const OpenMesh::Vec2f& _texcoord)
    {
      if (mesh.has_halfedge_texcoords2D())
        mesh.set_texcoord2D(_heh, _texcoord );
    }

    // edge attributes

    virtual void set_color(EdgeHandle h, const OpenMesh::Vec4uc& c)
    {
        if (mesh.has_edge_colors())
          mesh.set_color( h, Color( cc_from_uc(c[0]), cc_from_uc(c[1]), cc_from_uc(c[2]), cc_from_uc(c[3]) ) );
    }

    virtual void set_color(EdgeHandle h, const OpenMesh::Vec3uc& c)
    {
        if (mesh.has_edge_colors())
          mesh.set_color( h, Color( cc_from_uc(c[0]), cc_from_uc(c[1]), cc_from_uc(c[2]), 1.0 ) );
    }

    virtual void set_color(EdgeHandle h, const OpenMesh::Vec4f& c)
    {
        if (mesh.has_edge_colors())
          mesh.set_color( h, Color( c[0], c[1], c[2], c[3] ) );
    }

    virtual void set_color(EdgeHandle h, const OpenMesh::Vec3f& c)
    {
        if (mesh.has_edge_colors())
          mesh.set_color( h, Color( c[0], c[1], c[2], 1.0 ) );
    }

    // face attributes

    virtual void set_normal(FaceHandle h, const OpenMesh::Vec3f& n)
    {
      if (mesh.has_face_normals())
        mesh.set_normal( h, Normal( n[0], n[1], n[2] ) );
    }

    virtual void set_color(FaceHandle h, const OpenMesh::Vec3uc& c)
    {
      if (mesh.has_face_colors())
        mesh.set_color( h, Color( cc_from_uc(c[0]), cc_from_uc(c[1]), cc_from_uc(c[2]), 1.0 ) );
    }

    virtual void set_color(FaceHandle h, const OpenMesh::Vec4uc& c)
    {
      if (mesh.has_face_colors())
        mesh.set_color( h, Color( cc_from_uc(c[0]), cc_from_uc(c[1]), cc_from_uc(c[2]), cc_from_uc(c[3]) ) );
    }

    virtual void set_color(FaceHandle h, const OpenMesh::Vec3f& c)
    {
      if (mesh.has_face_colors())
        mesh.set_color( h, Color( c[0], c[1], c[2], 1.0 ) );
    }

    virtual void set_color(FaceHandle h, const OpenMesh::Vec4f& c)
    {
      if (mesh.has_face_colors())
        mesh.set_color( h, Color( c[0], c[1], c[2], c[3] ) );
    }

    virtual void add_face_texcoords( FaceHandle _fh, VertexHandle _vh, const std::vector<OpenMesh::Vec2f>& _face_texcoords)
    {
      // get first halfedge handle
      HalfedgeHandle cur_heh   = mesh.halfedge_handle(_fh);
      HalfedgeHandle end_heh   = mesh.prev_halfedge_handle(cur_heh);

      // find start heh
      while( mesh.to_vertex_handle(cur_heh) != _vh && cur_heh != end_heh )
        cur_heh = mesh.next_halfedge_handle( cur_heh);

      for(unsigned int i=0; i<_face_texcoords.size(); ++i)
      {
        set_texcoord( cur_heh, _face_texcoords[i]);
        cur_heh = mesh.next_halfedge_handle( cur_heh);
      }
    }

    virtual void set_face_texindex( FaceHandle _fh, int _texId ) {
      if ( mesh.has_face_texture_index() ) {
        mesh.set_texture_index(_fh , _texId);
      }
    }

    virtual void add_texture_information( int _id , std::string _name ) {
      OpenMesh::MPropHandleT< std::map< int, std::string > > property;

      if ( !mesh.get_property_handle(property,"TextureMapping") ) {
        mesh.add_property(property,"TextureMapping");
      }

      if ( mesh.property(property).find( _id ) == mesh.property(property).end() )
        mesh.property(property)[_id] = _name;
    }

    // low-level access to mesh

    virtual OpenMesh::BaseKernel* kernel() { return &mesh; }

    bool is_triangle_mesh() const
    { return triangular_mesh::is_triangles(); }

    void reserve(unsigned int nV, unsigned int nE, unsigned int nF)
    {
      mesh.reserve(nV, nE, nF);
    }

    // query number of faces, vertices, normals, texcoords
    size_t n_vertices()  const { return mesh.n_vertices(); }
    size_t n_faces()     const { return mesh.n_faces(); }
    size_t n_edges()     const { return mesh.n_edges(); }


    void prepare() { failed_faces_.clear(); }


    void finish()
    {
      if (!failed_faces_.empty())
      {
        omerr() << failed_faces_.size()
        << " faces failed, adding them as isolated faces\n";

        for (unsigned int i=0; i<failed_faces_.size(); ++i)
        {
          VHandles&  vhandles = failed_faces_[i];

          // double vertices
          for (unsigned int j=0; j<vhandles.size(); ++j)
          {
            Point p = mesh.point(vhandles[j]);
            vhandles[j] = mesh.add_vertex(p);
            // DO STORE p, reference may not work since vertex array
            // may be relocated after adding a new vertex !

            // Mark vertices of failed face as non-manifold
            if (mesh.has_vertex_status()) {
                mesh.status(vhandles[j]).set_fixed_nonmanifold(true);
            }
          }

          // add face
          FaceHandle fh = mesh.add_face(vhandles);

          // Mark failed face as non-manifold
          if (mesh.has_face_status())
              mesh.status(fh).set_fixed_nonmanifold(true);

          // Mark edges of failed face as non-two-manifold
          if (mesh.has_edge_status()) {
              typename triangular_mesh::FaceEdgeIter fe_it = mesh.fe_iter(fh);
              for(; fe_it.is_valid(); ++fe_it) {
                  mesh.status(*fe_it).set_fixed_nonmanifold(true);
              }
          }
        }

        failed_faces_.clear();
      }
    }
  private:

    triangular_mesh& mesh;
    std::vector<VHandles>  failed_faces_;
    // stores normals for halfedges of the next face
    std::map<VertexHandle,Normal> halfedgeNormals_;
  };

  triangular_mesh::triangular_mesh()
    : OpenMesh::TriMesh_ArrayKernelT< detail::triangular_mesh_traits > ()
  {}

  triangular_mesh::triangular_mesh( const std::string& filename )
  {
    load( filename );
  }

  bool
  triangular_mesh::load( const std::string& filename )
  {
    clear();
    ImporterT importer(*this);
    auto option = OpenMesh::IO::Options(); //todo: set correctly the options!
    if( OpenMesh::IO::IOManager().read( filename, importer,  option) )
      {
        update_normals();
        return true;
      }
    return false;
  }

  void
  triangular_mesh::save( const std::string& filename )
  {
    ExporterT exporter(*this);
    OpenMesh::IO::IOManager().write( filename, exporter );
  }

END_PROJECT_NAMESPACE

