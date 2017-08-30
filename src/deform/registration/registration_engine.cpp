#include "registration_engine.h"


RegistrationEngine::RegistrationEngine()
{
}
RegistrationEngine::~RegistrationEngine()
{
}

bool RegistrationEngine::initialize(const ConfigFile& )
{
    return false;
}
void RegistrationEngine::shutdown()
{
    
}

void RegistrationEngine::set_image_pair(int i, const Volume& fixed, const Volume& moving)
{
    i; fixed; moving;
}

void RegistrationEngine::build_pyramid()
{

}
