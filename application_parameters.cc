/*  Created on: Nov 19, 2015
 *      Author: T. Delame (tdelame@gmail.com)
 */
# include <application_parameters.h>
# include <sstream>
BEGIN_PROJECT_NAMESPACE 

  static const std::string input_word = "-i";
  static const std::string output_word = "-o";
  static const std::string velocity_word = "--wvelocity";
  static const std::string medial_word = "--wmedial";
  static const std::string iterations_word = "--iterations";
  static const std::string edge_length_word = "--edge";
  static const std::string zero_threshold_word = "--zero";
  static const std::string help_word = "--help";

  static const std::string help_message =
"MCF Curve Skeletonizer utility v0.1 ©2015 Thomas Delame\n\n"
"Usage: mcf_curve_skeletonizer " + input_word + " input_mesh_filename "
  + output_word + " output_filename\n"
"            [" + velocity_word + " velocity_weight]\n"
"            [" + medial_word + " medial_weight]\n"
"            [" + iterations_word + " nb_iterations]\n"
"            [" + edge_length_word + " edge_length_threshold]\n"
"            [" + zero_threshold_word + " zero_threshold]\n"
"            [" + help_word + "]\n\n"
"input_mesh_filename\n"
"\ta triangular surface mesh readable by OpenMesh\n\n"
"output_filename\n"
"\ta file that will contain the resulting skeleton graph in a JSON format."
"\tThis file can be used with the 3D skeleton web renderer (see README.md)\n\n"
"velocity_weight\n"
"\tthe weight associated to the velocity energy in the minimization process. It must be >0.\n"
"\tThe laplacian energy has a weight of 1.0\n\n"
"medial_weight\n"
"\tthe weight associated to the medial energy in the minimization process. It must be >0 \n\n"
"nb_iterations\n"
"\tnumber of times the algorithm will be applied. It must be >0"
"edge_length_threshold\n"
"\tthreshold for an edge length, under this threshold an edge is collapsed\n"
"zero_threshold\n"
"\tcurrently used to define an infinite weight by taking 1.0 / zero_threshold";

  static void help()
  {
    std::cout << help_message << std::endl;
  }

  application_parameters::application_parameters( int argc, char* argv[] )
    : wvelocity{ 0.1 }, wmedial{ 0.4 },
      edge_length_threshold{ -1.0}, zero_threshold{ 1e-6},
      iterations{ 5 }
  {
    enum{ NONE, INPUT, OUTPUT, VELOCITY, MEDIAL, ITERATION, EDGE_LENGTH, ZERO };
    int context = NONE;
    for( int i = 1; i < argc; ++ i )
      {
        auto word = std::string( argv[ i ] );
        if( context == NONE )
          {
            if( word == help_word )
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
            else if( word == edge_length_word )
              context = EDGE_LENGTH;
            else if( word == zero_threshold_word )
              context = ZERO;
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
        else if( context == EDGE_LENGTH )
          {
            std::istringstream tokenizer( word );
            tokenizer >> edge_length_threshold;
            if( tokenizer.fail() || edge_length_threshold < 0 )
              {
                LOG( fatal, "failed to recognize valid edge length threshold in argument " << word);
                help();
                exit( EXIT_FAILURE );
              }
            context = NONE;
          }
        else if( context == ZERO )
          {
            std::istringstream tokenizer( word );
            tokenizer >> zero_threshold;
            if( tokenizer.fail() || zero_threshold < 0 )
              {
                LOG( fatal, "failed to recognize valid zero threshold in argument " << word);
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

 END_PROJECT_NAMESPACE
