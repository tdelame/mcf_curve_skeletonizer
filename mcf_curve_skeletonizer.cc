/* Created on: Nov 17, 2015
 *     Author: T.Delame (tdelame@gmail.com)
 */
# include <project.h>
# include <mesh.h>
# include <json_writer.h>
# include <application_parameters.h>
# include <skeletonizer.h>

BEGIN_PROJECT_NAMESPACE

  skeletonizer::parameters
  build_skeletonizer_parameters(
      const application_parameters& params )
  {
    skeletonizer::parameters result;
    result.omega_velocity = params.wvelocity;
    result.omega_medial = params.wmedial;
    result.edge_length_threshold = params.edge_length_threshold;
    result.zero_threshold = params.zero_threshold;
    return result;
  }

  static int
  do_execute( int argc, char* argv[] )
  {
    application_parameters params( argc, argv );
    auto skeletonizer_parameters = build_skeletonizer_parameters( params );

    triangular_mesh mesh( params.input_mesh_filename );
    skeletonizer alg( mesh, skeletonizer_parameters );

    for( uint i = 0; i < params.iterations; ++ i )
      {
        alg.iterate();
      }
    alg.convert_to_skeleton();

    LOG( info, "MCF curve skeletonization with "
         << params.iterations << " iteration(s) done in "
         << alg.get_execution_duration() << " to produce a graph with "
         << mesh.n_vertices() << " vertices and " << mesh.n_edges() << " edges");

    json_writer( mesh, params.output_filename );

    return 0;
  }

END_PROJECT_NAMESPACE



int main( int argc, char* argv[] )
{
  return PROJECT_NAMESPACE::do_execute( argc, argv );
}
