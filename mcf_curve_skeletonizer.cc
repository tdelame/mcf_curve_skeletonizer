/* Created on: Nov 17, 2015
 *     Author: T.Delame (tdelame@gmail.com)
 */
# include <project.h>
# include <mesh.h>

# include <sstream>
BEGIN_PROJECT_NAMESPACE

  static const std::string input_word = "-i";
  static const std::string output_word = "-o";
  static const std::string velocity_word = "--wvelocity";
  static const std::string medial_word = "--wmedial";
  static const std::string iterations_word = "--iterations";

  static void
  help()
  {
    std::cout << "MCF Curve Skeletonizer utility v0.1 ©2015 Thomas Delame\n\n"
        << "Usage: mcf_curve_skeletonizer " << input_word << " input_mesh_filename "
        << output_word << " output_filename\n"
        << "            [" << velocity_word << " velocity_weight]\n"
        << "            [" << medial_word << " medial_weight]\n"
        << "            [" << iterations_word << " nb_iterations]\n"
        << "            [--help]\n\n"
        << "input_mesh_filename\n"
        << "\ta triangular surface mesh readable by OpenMesh\n\n"
        << "output_filename\n"
        << "\ta file that will contain the resulting mesh described in a mesh format supported by OpenMesh\n\n"
        << "velocity_weight\n"
        << "\tthe weight associated to the velocity energy in the minimization process. It must be >0.\n"
        << "\tThe laplacian energy has a weight of 1.0\n\n"
        << "medial_weight\n"
        << "\tthe weight associated to the medial energy in the minimization process. It must be >0 \n\n"
        << "nb_iterations\n"
        << "\tnumber of times the algorithm will be applied. It must be >0" << std::endl;
  }

  struct parameters {
    std::string input_mesh_filename;
    std::string output_filename;
    real wvelocity;
    real wmedial;
    uint iterations;


    explicit parameters( int argc, char* argv[] )
      : wvelocity{ 0.1 }, wmedial{ 0.4 }, iterations{ 5 }
    {
      enum{ NONE, INPUT, OUTPUT, VELOCITY, MEDIAL, ITERATION };
      int context = NONE;
      for( int i = 1; i < argc; ++ i )
        {
          auto word = std::string( argv[ i ] );
          if( context == NONE )
            {
              if( word == "--help")
                {
                  help();
                  exit( EXIT_SUCCESS );
                }
              else if( word == input_word )
                context = INPUT;
              else if( word == output_word )
                context = OUTPUT;
              else if( word == velocity_word )
                context = VELOCITY;
              else if( word == medial_word )
                context = MEDIAL;
              else if( word == iterations_word )
                context = ITERATION;
              else
                {
                  LOG(error,"argument " << word << " not recognized");
                }
            }
          else if( context == INPUT )
            {
              input_mesh_filename = word;
              context = NONE;
            }
          else if( context == OUTPUT )
            {
              output_filename = word;
              context = NONE;
            }
          else if( context == VELOCITY )
            {
              std::istringstream tokenizer( word );
              tokenizer >> wvelocity;
              if( tokenizer.fail() || wvelocity < 0 )
                {
                  LOG( fatal, "failed to recognize valid velocity weight in argument " << word);
                  help();
                  exit( EXIT_FAILURE );
                }
              context = NONE;
            }
          else if( context == MEDIAL )
            {
              std::istringstream tokenizer( word );
              tokenizer >> wmedial;
              if( tokenizer.fail() || wmedial < 0 )
                {
                  LOG( fatal, "failed to recognize valid medial weight in argument " << word);
                  help();
                  exit( EXIT_FAILURE );
                }
              context = NONE;
            }
          else if( context == ITERATION )
            {
              std::istringstream tokenizer( word );
              int result = -1;
              tokenizer >> result;
              if( tokenizer.fail() || result < 0 )
                {
                  LOG( fatal, "failed to recognize valid number of iterations in argument " << word );
                  help();
                  exit( EXIT_FAILURE );
                }
              iterations = uint(result);
              context = NONE;
            }
        }

      if( input_mesh_filename.empty() || output_filename.empty() )
        {
          LOG( fatal, "need to specify both the input and the input filenames");
          help();
          exit( EXIT_FAILURE );
        }
    }
  };

  static int
  do_execute( int argc, char* argv[] )
  {
    parameters params( argc, argv );

    triangular_mesh mesh( params.input_mesh_filename );

    LOG( info, "mesh information:");
    LOG( info, "\tnumber of vertices: " << mesh.n_vertices() );
    LOG( info, "\tnumber of edges   : " << mesh.n_edges() );
    LOG( info, "\tnumber of faces   : " << mesh.n_faces() );
    return 0;
  }

END_PROJECT_NAMESPACE



int main( int argc, char* argv[] )
{
  return PROJECT_NAMESPACE::do_execute( argc, argv );
}
