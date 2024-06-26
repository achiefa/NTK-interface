#include "NTK/AnalyticCostFunction.h"
#include "NTK/IterationCallBack.h"
#include "NTK/Timer.h"
#include "NTK/direxists.h"

#include <getopt.h>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <sstream>

double P10(double x) { return (1. / 256) * (46189 * pow(x, 10) - 109395 * pow(x, 8) + 90090 * pow(x, 6) - 30030 * pow(x, 4) + 3465 * pow(x, 2) - 63); }
int main(int argc, char *argv[])
{
    if ((argc - optind) != 2)
        {
          std::cerr << "Usage of " << argv[0] << ": <runcard> <result folder>" << std::endl;
          exit(-1);
        }

    namespace fs = std::filesystem; 

    // Input information
    const std::string InputCardPath = argv[optind];
    const std::string ResultFolder = argv[optind + 1];

    // Assign to the fit the name of the input card
    const std::string OutputFolder = ResultFolder + "/" + InputCardPath.substr(InputCardPath.find_last_of("/") + 1, InputCardPath.find(".yaml") - InputCardPath.find_last_of("/") - 1);
    if (!NTK::is_dir(ResultFolder))
    {
        printf("Folder %s does not exist. Creating a new one... \n", ResultFolder.c_str());
        fs::create_directories(ResultFolder);
    }

    if (NTK::is_dir(OutputFolder))
    {
        printf("A fit with name %s is already present in %s. Do you want to overwrite it? [(y)/n] \n", ResultFolder.c_str(), OutputFolder.c_str());

        int EXIT = 1;
        while(EXIT)
        {
            std::string buffer;
            std::getline(std::cin, buffer);
            if (buffer == "n")
            {
                printf("Interrupting data generation...\n");
                exit(-1);
            }
            else if (buffer.empty() || buffer == "y")
            {
                printf("Overwriting data file...\n");
                fs::remove(OutputFolder + "/Data.yaml");
                EXIT = 0;
            }
            else
            {
                printf("Choose a valid answer: [(y),n] \n");
            }
        }
    }
    else fs::create_directory(OutputFolder);

    if (!NTK::is_dir(OutputFolder + "/output"))
        fs::create_directories(OutputFolder + "/output");

    if (!NTK::is_dir(OutputFolder + "/log"))
        fs::create_directories(OutputFolder + "/log");

    // Copy input card into fit folder and rename it
    fs::copy(InputCardPath, OutputFolder, fs::copy_options::overwrite_existing);
    std::string FitName = InputCardPath.substr(InputCardPath.find_last_of("/") + 1, InputCardPath.find(".yaml") - InputCardPath.find_last_of("/") - 1);
    std::string oldname = OutputFolder + "/" + FitName + ".yaml";
    std::string newname = OutputFolder + "/meta.yaml";
    fs::rename(oldname, newname);

    // Read Input Card
    YAML::Node InputCard = YAML::LoadFile(InputCardPath);

    // ====================================================
    //          Data generation and fluctuations
    // ====================================================
    int n         = InputCard["Ndata"].as<int>();
    double xmin   = InputCard["xmin"].as<double>();
    double xmax   = InputCard["xmax"].as<double>();
    double yshift = InputCard["yshift"].as<double>();

    // noise
    double noise_mean = InputCard["noise_mean"].as<double>();
    double noise_sd = InputCard["noise_sd"].as<double>();
    int seed = InputCard["Seed"].as<int>();
    std::mt19937 rng;
    rng.seed(seed);
    std::normal_distribution<double> noise_gen(noise_mean, noise_sd);

    

    std::vector<NTK::dvec> truth;
    NTK::vecdata Data;
    for (int i = 0; i < n; i++)
    {   
        // TODO
        // Generalise this part for a general input vector x
        NTK::Datapoint tuple;
        double x = {xmin + i * (xmax - xmin) / n};
        double y = P10(x) + yshift;

        //_____
        // noise
        double noise = 0;
        while(! noise)
        if (y != 0)
            noise = noise_gen(rng)*y;
        else
            noise = noise_gen(rng);

        std::vector<double> xv(1,0.);
        std::vector<double> yvn(1,0.);
        std::vector<double> yv(1,0.);
        std::vector<double> noisev(1,0.);
        xv[0] = x;
        yv[0] = y;
        yvn[0] = y + noise;
        noisev[0] = std::abs(noise);
        
        std::get<0>(tuple) = xv;
        std::get<1>(tuple) = yvn;
        std::get<2>(tuple) = noisev;

        truth.push_back(yv);
        Data.push_back(tuple);
    }

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    //__________________________________
    emitter << YAML::Key << "Metadata";
    emitter << YAML::Value  << YAML::BeginMap;
    emitter << YAML::Key << "Number of points" << YAML::Value << n;
    emitter << YAML::Key << "Functionunction" << YAML::Value << "name of the function";
    emitter << YAML::Key << "xmin" << YAML::Value << xmin;
    emitter << YAML::Key << "xmax" << YAML::Value << xmax;
    emitter << YAML::Key << "Seed" << YAML::Value << seed;
    emitter << YAML::Key << "Noise distribution" << YAML::Value << "GAUSSIAN";
    emitter << YAML::Key << "Noise parameters";
    emitter << YAML::Value  << YAML::BeginMap;
    emitter << YAML::Key << "Mean of the noise" << YAML::Value << noise_mean;
    emitter << YAML::Key << "Std of the noise" << YAML::Value << noise_sd << YAML::EndMap;
    emitter << YAML::EndMap;
    emitter << YAML::Newline << YAML::Newline;
    emitter << YAML::Key << "Dependent variables";
    emitter << YAML::Value << YAML::BeginSeq;
    int id = 0;
    for (auto &d : Data)
    {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "Value" << YAML::Value << std::get<1>(d);
        emitter << YAML::Key << "Noise" << YAML::Value << std::get<2>(d);
        emitter << YAML::Key << "Truth" << YAML::Value << truth[id];
        emitter << YAML::EndMap;
        id++;
    }
    emitter << YAML::EndSeq;
    emitter << YAML::Newline << YAML::Newline;
    emitter << YAML::Key << "Independent variables";
    emitter << YAML::Value << YAML::BeginSeq;
    for (auto &d : Data)
    {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "Value" << YAML::Value << std::get<0>(d);
        emitter << YAML::EndMap;
    }
    //__________________________________
    emitter << YAML::EndMap;

    std::ofstream fout;
    fout = std::ofstream(OutputFolder + "/Data.yaml", std::ios::out | std::ios::app);
    fout << emitter.c_str();
    fout.close();
    
    return 0;
}