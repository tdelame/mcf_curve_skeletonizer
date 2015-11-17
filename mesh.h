/* Created on: Nov 17, 2015
 *     Author: T.Delame (tdelame@gmail.com)
 */
# ifndef PROJECT_MESH_H_
# define PROJECT_MESH_H_
# include <project.h>
# include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
BEGIN_PROJECT_NAMESPACE

  namespace detail {

    struct triangular_mesh_traits:
        public OpenMesh::DefaultTraits
    {
      typedef OpenMesh::Vec3d Point;
      typedef OpenMesh::Vec3d Normal;
      typedef OpenMesh::Vec4d Color;

      VertexAttributes   ( ( OpenMesh::Attributes::Normal
                         |   OpenMesh::Attributes::Status
                         |   OpenMesh::Attributes::Color        ) );
      HalfedgeAttributes ( ( OpenMesh::Attributes::PrevHalfedge
                         |   OpenMesh::Attributes::Status
                         |   OpenMesh::Attributes::Normal       ) );
      EdgeAttributes     (   OpenMesh::Attributes::Status         ); //very, very, VERY important: if not, you can't delete a face!
      FaceAttributes     ( ( OpenMesh::Attributes::Status
                         |   OpenMesh::Attributes::Normal
                         |   OpenMesh::Attributes::Color        ) );
    };
  }

  struct triangular_mesh:
      public OpenMesh::TriMesh_ArrayKernelT< detail::triangular_mesh_traits >
  {
    triangular_mesh();
    triangular_mesh( const std::string& filename );
    bool load( const std::string& filename );
    void save( const std::string& filename );
  };
END_PROJECT_NAMESPACE
# endif 
