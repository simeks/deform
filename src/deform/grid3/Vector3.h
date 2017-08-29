#ifndef VECTOR3
#define VECTOR3

#include <cmath>
#include <ostream>

//!Class representing 3D vectors, with elements of type T
template<class T>
class Vec3 {
public:
	T x;
	T y;
	T z;

public:
	//Standard constructor
	explicit Vec3(T _x=0, T _y=0, T _z=0) : x(_x), y(_y), z(_z) {}

	//Copy constructor
	template<class T2>
	Vec3(Vec3<T2> v): x(v.x), y(v.y), z(v.z) {}

	// vector addition
    Vec3<T> operator+(const Vec3<T> &v) const {
      return Vec3<T>(x+v.x,y+v.y,z+v.z);
    }

    Vec3<T>& operator+=(const Vec3<T> &v) {
      x += v.x;
      y += v.y;
      z += v.z;
      return (*this);
    }

	// vector subtraction
    Vec3<T> operator-(const Vec3<T> &v) const {
      return Vec3<T>(x-v.x,y-v.y,z-v.z);
    }

    Vec3<T>& operator-=(const Vec3<T> &v) {
      x -= v.x;
      y -= v.y;
      z -= v.z;
      return (*this);
    }

	Vec3<T> operator*(T d) const {
		return Vec3<T>(x*d, y*d, z*d);
	}

    Vec3<T>& operator*=(const double &d) {
      x = T(x*d);
      y = T(y*d);
      z = T(z*d);
      return (*this);
    }

	// division with double
    Vec3<T> operator/(const double &d) const {
      return Vec3<T>(T(x/d),T(y/d),T(z/d));
    }

	Vec3<T>& operator/=(const double &d) {
      x = T(x/d);
      y = T(y/d);
      z = T(z/d);
      return (*this);
    }

	// scalar product
    double operator*(const Vec3<T> &v) const {
      return x*v.x + y*v.y + z*v.z;
    }

	// cross product
    Vec3<T> cross(const Vec3<T> &v) const {
      Vec3<T> tmp;
      tmp.x=y*v.z-z*v.y;
      tmp.y=z*v.x-x*v.z;
      tmp.z=x*v.y-y*v.x;
      return tmp;
    }

	// Element wise multiplication (hadamard product)
	 Vec3<T> operator%(const Vec3<T> &v) const {
      return Vec3<T>(x*v.x, y*v.y, z*v.z);
    }

	// normalize
    Vec3<T>& normalize() {
      double n = norm();
      (*this) /= n;
      return (*this);
    }

	// vector norm
    double norm() const {
      return std::sqrt(double(x*x+y*y+z*z));
    }
    
    // vector norm squared
    double norm2() const {
      return x*x+y*y+z*z;
    }

	//! equal (used for integer position vectors)
    bool operator==(const Vec3<T> &v) const {
      return (x==v.x && y==v.y && z==v.z);
    }

    //! not equal (used for integer position vectors)
    bool operator!=(const Vec3<T> &v) const {
      return !((*this)==v);
    }

	 //! multiplication with double
    friend Vec3<T> operator*(const double &d, const Vec3<T> &v) {
      return v*d;
    }


	//!Write to ostream
	friend std::ostream& operator<<(std::ostream &os, const Vec3<T> &v) {
      os << v.x << " " << v.y<<" "<<v.z;
      return os;
    }



};



template class Vec3<float>;
template class Vec3<double>;
template class Vec3<int>;

typedef Vec3<float>     Vec3f;
typedef Vec3<double>    Vec3d;
typedef Vec3<int>       Vec3i;

#endif
