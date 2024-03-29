#include <sys/stat.h>
#include <string>

namespace NTK
{
    bool is_dir(const std::string &dir)
    {
        struct stat buffer;
        
        if (stat(dir.c_str(), &buffer) == 0)
            return true;
        else
            return false;
    }
}