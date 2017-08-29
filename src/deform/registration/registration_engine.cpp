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

bool RegistrationEngine::set_fixed_image(int , const Grid3<element_type>& )
{
    return false;
}
bool RegistrationEngine::set_moving_image(int , const Grid3<element_type>& )
{
    return false;
}

void RegistrationEngine::build_pyramid()
{

}
