/*  Created on: Nov 19, 2015
 *      Author: T. Delame (tdelame@gmail.com)
 */

# ifndef ATUIN_APPLICATION_PARAMETERS_H_
# define ATUIN_APPLICATION_PARAMETERS_H_
# include <project.h>

BEGIN_PROJECT_NAMESPACE 

  struct application_parameters {
    std::string input_mesh_filename;
    std::string output_filename;
    real wvelocity;
    real wmedial;
    real edge_length_threshold;
    real zero_threshold;
    uint iterations;

    explicit application_parameters( int argc, char* argv[] );
  };

END_PROJECT_NAMESPACE

#endif /* ATUIN_APPLICATION_PARAMETERS_H_ */
