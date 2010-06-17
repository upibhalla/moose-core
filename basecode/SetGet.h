/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SETGET_H
#define _SETGET_H

class SetGet
{
	public:
		SetGet( const Eref& e )
			: e_( e )
		{;}

		virtual ~SetGet()
		{;}

		/**
		 * Assigns 'field' on 'tgt' to 'val', after doing necessary type
		 * conversion from the string val. Returns 0 if it can't
		 * handle it, which is unusual.
		 */
		virtual bool innerStrSet( const Eref& tgt, 
			const string& field, const string& val ) const = 0;

		/**
		 * Looks up field value on tgt, converts to string, and puts on 
		 * 'ret'. Returns 0 if the class does not support 'get' operations,
		 * which is usually the case. Fields are the exception.
		 */
		virtual bool innerStrGet( const Eref& tgt, 
			const string& field, string& ret ) const {
			return 0;
		}


		/**
		 * Checks arg # and types for a 'set' call. Can be zero to 3 args.
		 * Returns true if good. Passes back found fid.
		 * Utility function to check that the target field matches this
		 * source type, and to look up and pass back the fid.
		 */
		bool checkSet( const string& field, Eref& tgt, FuncId& fid ) const;

//////////////////////////////////////////////////////////////////////
		/**
		 * Blocking 'get' call, returning into a string.
		 * There is a matching 'get<T> call, returning appropriate type.
		 */
		static bool strGet( const Eref& tgt, const string& field, string& ret );

		/**
		 * Blocking 'set' call, using automatic string conversion
		 * There is a matching blocking set call with typed arguments.
		 */
		static bool strSet( const Eref& dest, const string& field, const string& val );

		
		/**
		 * Waits for completion of a nonblocking 'set' call, either
		 * string or typed versions.
		 * Can be skipped if there is an absolute guarantee that there
		 * won't be dependencies between the 'set' and subsequent calls,
		 * till the next completeSet or harvestGet call.
		 * Avoid using. If you have dependencies then use the blocking set.
		 */
		void completeSet() const;

		char* buf();

	private:
		Eref e_;
};

class SetGet0: public SetGet
{
	public:
		SetGet0( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field )
		{
			SetGet0 sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( sg.checkSet( field, tgt, fid ) ) {
				Shell::dispatchSet( tgt, fid, "", 0 );
				/*
				sg.iSetInner( fid, "", 0 );

				// Ensure that clearQ is called before this return.
				sg.completeSet();
				*/
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const
		{
			return set( dest, field );
		}
};

template< class A > class SetGet1: public SetGet
{
	public:
		SetGet1( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, A arg )
		{
			SetGet1< A > sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A > conv( arg );
				char *temp = new char[ conv.size() ];
				conv.val2buf( temp );
				Shell::dispatchSet( tgt, fid, temp, conv.size() );
				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * I would like to provide a vector set operation. It should
		 * work in three modes: Set all targets to the same value,
		 * set targets one by one to a vector of values, and set targets
		 * one by one to randomly generated values within a range. All
		 * of these can best be collapsed into the vector assignment 
		 * operation.
		 * This variant requires that all vector entries have the same
		 * size. Strings won't work.
		 */
		static bool setVec( const Eref& dest, const string& field, 
			const vector< A >& arg )
		{
			SetGet1< A > sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( arg.size() == 0 )
				return 0;

			if ( sg.checkSet( field, tgt, fid ) ) {
				const char* data = reinterpret_cast< const char* >( &arg[0] );
				PrepackedBuffer pb( data, arg.size() * sizeof( A ), 
					arg.size() ) ;
				Shell::dispatchSetVec( tgt, fid, pb );
				return 1;
			}
			return 0;
		}

		/**
		 * Sets all target array values to the single value
		 */
		static bool setRepeat( const Eref& dest, const string& field, 
			const A& arg )
		{
			vector< A >temp ( 1, arg );
			return setVec( dest, field, temp );
		}

		/**
		 * Blocking call using string conversion
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const
		{
			A arg;
			Conv< A >::str2val( arg, val );
			return set( dest, field, arg );
		}

		/**
		 * Nonblocking 'set' call, using automatic string conversion into
		 * arbitrary numbers of arguments.
		 * There is a matching nonblocking set call with typed arguments.
		 */
		bool iStrSet( const string& field, const string& val )
		{
			A temp;
			Conv< A >::str2val( temp, val );
			return iSet( field, temp );
		}
};

template< class A > class Field: public SetGet1< A >
{
	public:
		Field( const Eref& dest )
			: SetGet1< A >( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, A arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::set( dest, temp, arg );
		}

		static bool setVec( const Eref& dest, const string& field, 
			const vector< A >& arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::setVec( dest, temp, arg );
		}

		static bool setRepeat( const Eref& dest, const string& field, 
			A arg )
		{
			string temp = "set_" + field;
			return SetGet1< A >::setRepeat( dest, temp, arg );
		}

		/**
		 * Blocking call using string conversion
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const 
		{
			A arg;
			str2val( arg, val );
			return set( dest, field, arg );
		}

	//////////////////////////////////////////////////////////////////

		/**
		 * Blocking call using typed values
		 */
		static A get( const Eref& dest, const string& field)
		{ 
			SetGet1< A > sg( dest );
			string temp = "get_" + field;
			const char* ret = Shell::dispatchGet( dest, temp, &sg );
			Conv< A > conv( ret );
			return *conv;
		}

		/**
		 * Blocking virtual call for finding a value and returning in a
		 * string.
		 */
		bool innerStrGet( const Eref& dest, const string& field, 
			string& str ) const
		{
			SetGet1< A > sg( dest );
			string temp = "get_" + field;
			const char* ret = Shell::dispatchGet( dest, temp, &sg );
			Conv< A > conv( ret );
			val2str( str, *conv );
			return 1;
		}
};

/**
 * SetGet2 handles 2-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2 > class SetGet2: public SetGet
{
	public:
		SetGet2( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2 )
		{
			SetGet2< A1, A2 > sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				char *temp = new char[ conv1.size() + conv2.size() ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				Shell::dispatchSet( tgt, fid, temp, 
					conv1.size() + conv2.size() );
				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const
		{
			cout << "innerStrSet< A1, A2 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2 );
		}

	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * SetGet3 handles 3-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3 > class SetGet3: public SetGet
{
	public:
		SetGet3( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3 )
		{
			SetGet3< A1, A2, A3 > sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				unsigned int s1 = conv1.size();
				unsigned int s1s2 = s1 + conv2.size();
				unsigned int totSize = s1s2 + conv3.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + s1 );
				conv3.val2buf( temp + s1s2 );
				Shell::dispatchSet( tgt, fid, temp, totSize );
				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const 
		{
			cout << "innerStrSet< A1, A2, A3 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2, arg3 );
		}

	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * SetGet4 handles 4-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3, class A4 > class SetGet4: public SetGet
{
	public:
		SetGet4( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4 )
		{
			SetGet4< A1, A2, A3, A4 > sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				unsigned int s1 = conv1.size();
				unsigned int s1s2 = s1 + conv2.size();
				unsigned int s1s2s3 = s1s2 + conv3.size();
				unsigned int totSize = s1s2s3 + conv4.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + s1 );
				conv3.val2buf( temp + s1s2 );
				conv4.val2buf( temp + s1s2s3 );
				Shell::dispatchSet( tgt, fid, temp, totSize );
				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const
		{
			cout << "innerStrSet< A1, A2, A3, A4 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2, arg3, arg4 );
		}

	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

/**
 * SetGet5 handles 5-argument Sets. It does not deal with Gets.
 */
template< class A1, class A2, class A3, class A4, class A5 > class SetGet5:
	public SetGet
{
	public:
		SetGet5( const Eref& dest )
			: SetGet( dest )
		{;}

		/**
		 * Blocking, typed 'Set' call
		 */
		static bool set( const Eref& dest, const string& field, 
			A1 arg1, A2 arg2, A3 arg3, A4 arg4, A5 arg5 )
		{
			SetGet5< A1, A2, A3, A4, A5 > sg( dest );
			FuncId fid;
			Eref tgt( dest );
			if ( sg.checkSet( field, tgt, fid ) ) {
				Conv< A1 > conv1( arg1 );
				Conv< A2 > conv2( arg2 );
				Conv< A3 > conv3( arg3 );
				Conv< A4 > conv4( arg4 );
				Conv< A5 > conv5( arg5 );
				unsigned int totSize = conv1.size() + conv2.size() +
					conv3.size() + conv4.size() + conv5.size();
				char *temp = new char[ totSize ];
				conv1.val2buf( temp );
				conv2.val2buf( temp + conv1.size() );
				conv3.val2buf( temp + conv1.size() + conv2.size() );
				conv4.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() );
				conv5.val2buf( temp + conv1.size() + conv2.size() + 
					conv3.size() + conv4.size() );
				Shell::dispatchSet( tgt, fid, temp, totSize );

				delete[] temp;
				return 1;
			}
			return 0;
		}

		/**
		 * Blocking call using string conversion.
		 * As yet we don't have 2 arg conversion from a single string.
		 * So this is a dummy
		 */
		bool innerStrSet( const Eref& dest, const string& field, 
			const string& val ) const
		{
			cout << "innerStrSet< A1, A2, A3, A4, A5 >: string convertion not yet implemented\n";
			A1 arg1;
			A2 arg2;
			A3 arg3;
			A4 arg4;
			A5 arg5;
			str2val( arg1, val );
			return set( dest, field, arg1, arg2, arg3, arg4, arg5 );
		}
	//////////////////////////////////////////////////////////////////
	//  The 'Get' calls for 2 args are currently undefined.
	//////////////////////////////////////////////////////////////////
	
		/**
		 * Terminating call using string conversion
		 */
		string harvestStrGet() const
		{ 
			return "";
		}
};

#endif // _SETGET_H
