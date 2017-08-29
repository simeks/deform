#pragma once

typedef float element_type;

class ConfigFile;
class Optimizer;
class Volume;

class RegistrationEngine
{
public:
    RegistrationEngine();
    ~RegistrationEngine();

    bool initialize(const ConfigFile& cfg);
    void shutdown();

    bool set_image_pair(int i, const Volume& fixed, const Volume& moving);
    
private:
    void build_pyramid();


};