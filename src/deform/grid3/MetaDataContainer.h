#ifndef METADATA_CONTAINER
#define METADATA_CONTAINER

#include <map>
using namespace std;

//!Class for storing meta data related to a volume image
class MetaDataContainer{
private:
	map<string, double> double_data;
	map<string, long> long_data;
	map<string, bool> bool_data;
	map<string, string> string_data;

public:
	void SetDouble(string key, double value){
		double_data[key]=value;
	}

	double GetDouble(string key, double default=0){
		auto it=double_data.find(key);
		if(it!=double_data.end()){
			return it->second;
		}
		return default;
	}

	void SetLong(string key, long value){
		long_data[key]=value;
	}

	long GetLong(string key, long default=0){
		auto it=long_data.find(key);
		if(it!=long_data.end()){	
			return it->second;
		}
		return default;
	}

	void SetBool(string key, bool value){
		bool_data[key]=value;
	}

	bool GetBool(string key, bool default=false){
		auto it=bool_data.find(key);
		if(it!=bool_data.end()){	
			return it->second;
		}
		return default;
	}

	void SetString(string key, string value){
		string_data[key]=value;
	}

	string GetString(string key, string default=""){
		auto it=string_data.find(key);
		if(it!=string_data.end()){	
			return it->second;
		}
		return default;
	}

};


#endif
