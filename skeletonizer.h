/* Created on: Nov 17, 2015
 *     Author: T.Delame (tdelame@gmail.com)
 */
# ifndef PROJECT_SKELETONIZER2_H_
# define PROJECT_SKELETONIZER2_H_

# include <project.h>
# include <mesh.h>
# include <skeletonizer.h>
BEGIN_PROJECT_NAMESPACE

  class skeletonizer {
  public:
    struct parameters {
      real omega_velocity;
      real omega_medial;
      real edge_length_threshold;
      real zero_threshold;
      real inv_zero_threshold;

      parameters();
    };


    skeletonizer(
        triangular_mesh& mesh,
        const parameters& params = parameters{} );

    // one step of the algorithm
    void
    iterate();

    // post-processing step: convert the input mesh to a curve skeleton (a graph)
    void
    convert_to_skeleton();

    // return the total time spent in iterate() and convert_to_skeleton()
    real
    get_execution_duration() const noexcept;

  private:
    typedef triangular_mesh::VertexHandle vhandle;
    typedef triangular_mesh::EdgeHandle ehandle;
    typedef triangular_mesh::HalfedgeHandle hehandle;
    typedef triangular_mesh::FaceHandle fhandle;
    typedef triangular_mesh::Point point;

    // set up variable, including the computation of medial point cloud
    void
    initialize();

    // energy minimization to contract the geometry
    void
    contract_geometry();

    // collapse and split operations to update the topology
    void
    update_topology();

    // fix degenerated portions of the input mesh
    void
    detect_degeneracies();

    inline real&
    cotangent_weight( const ehandle& handle )
    {
      return mesh.property( cotangent_weight_handle, handle );
    }

    inline real&
    cotangent_weight( const hehandle& handle )
    {
      return mesh.property( cotangent_weight_handle, ehandle{ handle.idx() >> 1 } );
    }

    inline real&
    laplacian_weight( const vhandle& handle )
    {
      return mesh.property( omega_L_handle, handle );
    }

    inline real&
    velocity_weight( const vhandle& handle )
    {
      return mesh.property( omega_V_handle, handle );
    }

    inline real&
    medial_weight( const vhandle& handle )
    {
      return mesh.property( omega_M_handle, handle );
    }

    inline point&
    medial_point( const vhandle& handle )
    {
      return mesh.property( medial_handle, handle );
    }

    inline uint8_t&
    status( const vhandle& handle )
    {
      return mesh.property( status_handle, handle );
    }

    inline real&
    halpha( const hehandle& handle )
    {
      return mesh.property( alpha_handle, handle );
    }

    void
    fix( const vhandle& handle );

    OpenMesh::EPropHandleT< real > cotangent_weight_handle;
    OpenMesh::HPropHandleT< real > alpha_handle;
    OpenMesh::VPropHandleT< real > omega_L_handle;
    OpenMesh::VPropHandleT< real > omega_V_handle;
    OpenMesh::VPropHandleT< real > omega_M_handle;
    OpenMesh::VPropHandleT< point > medial_handle;
    OpenMesh::VPropHandleT< uint8_t > status_handle;

    triangular_mesh& mesh;
    parameters params;
    real duration;
  };


END_PROJECT_NAMESPACE
# endif 
