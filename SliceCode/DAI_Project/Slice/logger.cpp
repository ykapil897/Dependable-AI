#include "slice.h"

std::ofstream Logger::log_file;
bool Logger::initialized = false;
bool Logger::verbose = true;
std::string Logger::log_path = "";
std::mutex Logger::log_mutex;
bool Logger::console_output_enabled = false; 
int Logger::verbosity_level = 1; 