/*  Created on: Nov 19, 2015
 *      Author: T. Delame (tdelame@gmail.com)
 */

# ifndef ATUIN_JSON_WRITER_H_
# define ATUIN_JSON_WRITER_H_
# include <project.h>
# include <mesh.h>
# include <fstream>
BEGIN_PROJECT_NAMESPACE 
  class triangular_mesh;
  class json_writer {
  public:
    json_writer(
       triangular_mesh& input,
       const std::string& filename );
  private:
    void info();
    void atoms();
    void topology();

    triangular_mesh& input;
    std::string filename;
    std::ofstream output;
 };
 END_PROJECT_NAMESPACE

#endif /* ATUIN_JSON_WRITER_H_ */
