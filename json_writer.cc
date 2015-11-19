/*  Created on: Nov 19, 2015
 *      Author: T. Delame (tdelame@gmail.com)
 */
# include <json_writer.h>
# include <mesh.h>
BEGIN_PROJECT_NAMESPACE
  json_writer::json_writer( triangular_mesh& input, const std::string& filename ) :
    input{ input }, filename{ filename }, output{ filename }
  {
    output.precision( 5 );
    output << "{";
    info();
    atoms();
    topology();
    output << "}" << std::flush;
    output.close();
  }

  void
  json_writer::info()
  {
    //for now, there is no radii. I put radii between 0.0 and 0.1 in order
    //to have enough variation to compute atom colors for the webgl renderer.
    real min_radius = 0.0;
    real max_radius = 0.1;
    auto atoms = input.n_vertices();
    auto links = input.n_edges();
    output << "\"author\":\"T.Delame\","
          << "\"number_of_atoms\":" << atoms << ","
          << "\"number_of_links\":" << links << ","
          << "\"max_radius\":" << max_radius << ","
          << "\"min_radius\":" << min_radius << ",";
  }

  void
  json_writer::atoms()
  {
    output << "\"atoms\":[";
    auto atoms = input.n_vertices();
    const auto step = real(0.1 / atoms);
    for( uint32_t i = 0; i < atoms; ++ i )
     {
       const auto point = input.point( triangular_mesh::VertexHandle(i) );
       output << point[0] << ',' << point[1] << ',' << point[2] << ',' << i * step;
       if( i + 1 < atoms ) output << ',';
     }
    output << "],";
  }

  void
  json_writer::topology( )
  {
    output << "\"links\":[";
    auto links = input.n_edges();
    for( uint64_t i = 0; i < links; ++ i )
      {
        if( i ) output << ',';
        auto he = triangular_mesh::HalfedgeHandle( i << 1 );
        output << input.to_vertex_handle( he ).idx() << ','
            << input.from_vertex_handle( he ).idx();
       }
     output << ']';
   }





END_PROJECT_NAMESPACE
