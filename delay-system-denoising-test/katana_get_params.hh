#ifndef KATANA_GET_PARAMS__
#define KATANA_GET_PARAMS__			//include guard

/* A small and light-weight headers-only implementation for defining a command line name of a variable to make it readable from the command line 

Written by C. Redlich at TU Berlin in... probably 2014 or something. */


/* 
Usage example: 

double MyDouble = katana::getCmdOption(argv, argv + argc, "-MyDouble", 1.5)

In the claisscal c++ main the first two arguments don't have to be changed by the user. The third is the command line command that you want to use, while the last is the standard value. 

This will allow you to change your variable called MyDouble via the typical command line interface: 
'[ProgramName] -MyDouble 3.1415' 
will set the value to 3.1415

If no command line argument is parsed, the default value (1.5 in the example) is used. 

Be aware: This class distinguishes between parsing a double or an int from the command line by what you set the standard value to! If you want a double, but the standard value to be 0, set it to 0.0!*/


#include <algorithm>
#include <string>
#include <cstdlib>
#include <iostream>

namespace  katana
{

	double getCmdOption(const char ** begin, const char ** end, const std::string & option, double Default)
	{
		const char ** itr = std::find(begin, end, option);
		if (itr != end && ++itr != end)
		{
		char* b;
		double d;
		d = strtod(*itr, &b);
		if (0 == d && *itr == b) 
			{	std::cout << "Input of Option ''" << option << "'' was wrong. Setting to default: " << option << " " << Default << std::endl;      // error handling.
				return Default;
			}
		std::cout << "Set Option: "<< option << " " << d << std::endl;
			return d;
	
		}
		return Default;
	}

	int getCmdOption(const char ** begin, const char ** end, const std::string & option, int Default)
	{
		const char ** itr = std::find(begin, end, option);
		if (itr != end && ++itr != end)
		{
		//char* b;
		int d;
		d = atoi(*itr);
		//if (0 == d && *itr == b) 
			//{	std::cout << "Input of Option ''" << option << "'' was wrong. Setting to default: " << option << " " << Default << std::endl;      // error handling.
				//return Default;
			//}
		std::cout << "Set Option: "<< option << " " << d << std::endl;
			return d;
	
		}
		return Default;
	}

	
	std::string getCmdOption(const char ** begin, const char ** end, const std::string & option, std::string Default)
	{
		const char ** itr = std::find(begin, end, option);
		if (itr != end && ++itr != end)
		{
		std::cout << "Set Option: "<< option << " = " << *itr << std::endl;
			return *itr;
	
		}
		return Default;
	}
	
	
	bool getCmdOption_bool(const char ** begin, const char ** end, const std::string & option, bool Default)
	{
		const char ** itr = std::find(begin, end, option);
		if (itr != end)
		{	bool Val=!Default;
		std::cout << "Set Option: "<< option << " = " << Val << std::endl;
			return Val;
	
		}
		return Default;
	}
	
	
	
	bool cmdOptionExists(const char** begin, const char** end, const std::string& option)
	{
		return std::find(begin, end, option) != end;
	}

}

#endif //End of include guard for KATANA_GET_PARAMS__
