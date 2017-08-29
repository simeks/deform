#pragma once


#include "Vector3.h"
#include "MetaDataContainer.h"
#include <algorithm>
#include <string>
#include <stdint.h>
#include <vector>
using namespace std;

/*! \mainpage Grid3 documentation
 *
 * \section intro_sec Introduction
 *
 * This is the documentation for the Grid3 library. This documentation was generated using Doxygen.
 *
 * \section usage_sec Using Grid3
 *
 * To use the Grid3 library in your C++ project, simply include the Grid3.h header file, and point your compiler to the ".../Grid3/include" folder.
 *  
 */


//!Class representing a 3D image grid, each voxel being of type T
template<class T>
class Grid3{

protected:
	std::vector<T> data;
	int _width, _height, _depth;
	double _spacingX, _spacingY, _spacingZ;
	double _originX, _originY, _originZ;
	MetaDataContainer metadata;

public:
	//!Default constructor. 
	Grid3(){}

	//!Construct a grid with given dimensions.
	Grid3(int width, int height, int depth=1, double x_spacing=1.0, double y_spacing=1.0, double z_spacing=1.0);

	//!Construct a grid with given dimensions.
	Grid3(Vec3i dimensions, Vec3d spacing=Vec3d(1.0,1.0,1.0), Vec3d origin=Vec3d(0.0,0.0,0.0));

	//!Initialize the grid with given dimensions. Any previous data in the grid will be overwritten.
	void Init(int width, int height, int depth=1,  double x_spacing=1.0, double y_spacing=1.0, double z_spacing=1.0); 

	//!Return the number of grid columns. 
	int GetWidth(){ return _width;}

	//!Return the number of grid rows. 
	int GetHeight(){ return _height;}

	//!Return the number of grid slices. 
	int GetDepth(){ return _depth; }

	//!Return a vector containing the grid dimensions
	Vec3i GetDimensions(){return Vec3i(_width, _height, _depth);}

	//!Return the spacing along X. 
	double GetSpacingX(){return _spacingX;}

	//!Return the spacing along Y. 
	double GetSpacingY(){return _spacingY;}

	//!Return the spacing along Z. 
	double GetSpacingZ(){return _spacingZ;}

	//!Return the spacing along all three dimesions. 
	Vec3d GetSpacing(){return Vec3d(_spacingX, _spacingY, _spacingZ);}

	Vec3d GetOrigin() const { return Vec3d(_originX, _originY, _originZ); }

	//!Set spacing.
	void SetSpacing(double x_spacing, double y_spacing, double z_spacing);

	//!Set spacing.
	void SetSpacing(Vec3d spacing);

	//!Transform point p from world coordinates to grid coordinates.
	Vec3d WorldToGrid(Vec3d p);

	//!Transform point p from grid coordinates to world coordinates.
	Vec3d GridToWorld(Vec3d p);


	//!Returns a MetaDataContainer with the current image meta data.
	MetaDataContainer GetMetaData(){return metadata;}

	//!Set image meta data. All current meta data is overwritten. 
	void SetMetaData(MetaDataContainer md){metadata=md;}

	//!Return the total number of grid points. 
	int NumberOfElements();

	//!Returns a pointer to the beginning of the internal data array. The called object owns the pointer.
	T* Data();

	//!Return a string representation of the voxel type
	std::string GetDataType();
	 
	//!Returns the index of a specified coordinate in the internal data array.
	int Offset(int x, int y, int z);

	//!Returns the index of a specified coordinate in the internal data array.
	int Offset(Vec3i p){ return Offset(p.x,p.y,p.z); }

	//!Returns the coordinate corresponding to a specified index in the internal data array.
	Vec3i Coordinate(long index);

	//!Test if the given grid point is within the bounds of the grid. 
	bool IsInside(int x, int y, int z);

	//!Test if the given grid point is within the bounds of the grid. 
	bool IsInside(Vec3i p);

	//!Access to the grid value at a specified index. The passed index is assumed to be within the grid bounds.
	T& operator[](long index);

	//!Access to the grid value at a specified coordinate. If "Safe" is false, the passed coordinate is assumed to be within the grid bounds. If "Safe" is true, a reference to the closest grid point is returned for out-of-bounds coordinates.
	T& operator()(int x,int y, int z=0, bool Safe=false);

	//!Access to the grid value at a specified coordinate. If "Safe" is false, the passed coordinate is assumed to be within the grid bounds. If "Safe" is true, a reference to the closest grid point is returned for out-of-bounds coordinates.
	T& operator()(Vec3i p, bool Safe=false);

	//!Get the grid value at a specified coordinate using trilinear interpolation.
	T LinearAt(double x, double y, double z, bool border = false);

	//!Get the grid value at a specified coordinate using trilinear interpolation.
	T LinearAt(Vec3d p, bool border = false);

	//!Fill the vector n with the indices of the 6-neighbors of the gridpoint at the specified index
	void Get6Neighbors(long index, vector<long>& n);
	
	//!Fill the vector n with the coordinates of the 6-neighbors of the gridpoint at the specified coordinate
	void Get6Neighbors(Vec3i p, vector<Vec3i>& n);

	//!Set all points in the grid to a specified value.
	void Fill(T value);

	//!Return the maximum value in the grid
	T GetMaxValue();

	//!Return the minimum value in the grid
	T GetMinValue();

	/*
	//!Pointwise multiplication by scalar
	Grid3<T> operator*(double x);
		
	//!Pointwise division by scalar
	Grid3<T> operator/(double x);

	//!Pointwise addition
	Grid3<T> operator+(T x);

	//!Pointwise subtraction
	Grid3<T> operator-(T x);*/

	//! Replace the current grid data with that of another grid. If the two grids do not have the same NumberOfElements(), nothing happens.
	void CopyDataFrom(Grid3<T>& src);

	//! Swap the contents of two grids. If the two grids do not have the same NumberOfElements(), nothing happens.
	void SwapContentsWith(Grid3<T>& v);

	//!Construct grid from an image file
	bool CreateFromFile(std::string filename); 

	//!Construct grid from an image file in VTK legacy format
	bool ReadVTK(std::string filename);

	//!Save grid in VTK legacy format
	bool WriteVTK(std::string filename);

};
