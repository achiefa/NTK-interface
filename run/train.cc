#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"
#include "NTK/Timer.h"
#include "NTK/direxists.h"

#include <getopt.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
  // SET A FLAG TO STOP AT INITIALISATION
  if ((argc - optind) != 4)
    {
      std::cerr << "Usage of " << argv[0] << ": <replica index> <runcard> <data folder> <result folder>" << std::endl;
      exit(-1);
    }

  // Input information
  int replica = atoi(argv[optind]);
  const std::string InputCardPath = argv[optind + 1];
  const std::string DataFolder = argv[optind + 2];
  const std::string ResultFolder = argv[optind + 3];

  // Assign to the fit the name of the input card
  const std::string OutputFolder = ResultFolder + "/" + InputCardPath.substr(InputCardPath.find_last_of("/") + 1, InputCardPath.find(".yaml") - InputCardPath.find_last_of("/") - 1);

  // Require that the result folder exists. If not throw an exception.  
  if (NTK::is_dir(ResultFolder))
    printf("Folder %s does not exist. Creating a new one... \n", ResultFolder.c_str());

  // Read Input Card
  YAML::Node config = YAML::LoadFile(InputCardPath);

  // Set the seeds based on replica wanted
  //config["NNAD"]["seed"] = config["NNAD"]["seed"].as<int>() + replica;

  return 0;
}