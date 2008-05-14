/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../element/Neutral.h"
#include "DeletionMarkerFinfo.h"
#include "GlobalMarkerFinfo.h"
#include "ThisFinfo.h"

/**
 * This sets up initial space on each ArrayElement for 4 messages.
 * We expect always to see the parent, process and usually something else.
 */
static const unsigned int INITIAL_MSG_SIZE = 4;

#ifdef DO_UNIT_TESTS
int ArrayElement::numInstances = 0;
#endif

ArrayElement::ArrayElement(
				Id id,
				const std::string& name, 
				void* data,
				unsigned int numSrc, 
				unsigned int numEntries,
				size_t objectSize
	)
	: Element( id ), name_( name ), 
		data_( data ), 
		msg_( numSrc ), 
		numEntries_(numEntries), 
		objectSize_(objectSize)
{
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif	
		;
}

ArrayElement::ArrayElement( const std::string& name, 
			const vector< Msg >& msg, 
			const map< int, vector< ConnTainer* > >& dest,
			const vector< Finfo* >& finfo, 
			void *data, 
			int numEntries, 
			size_t objectSize
		): Element (Id::scratchId()), name_(name), msg_(msg), dest_(dest), 
			finfo_(finfo), data_(data), numEntries_(numEntries), objectSize_(objectSize)
		{
		;
		}

/**
 * Copies a ArrayElement. Does NOT copy data or messages.
 */
ArrayElement::ArrayElement( const ArrayElement* orig )
		: Element( Id::scratchId() ),
		name_( orig->name_ ), 
		finfo_( 1 ),
		data_( 0 ),
		msg_( orig->cinfo()->numSrc() )
{
	assert( finfo_.size() > 0 );
	// Copy over the 'this' finfo
	finfo_[0] = orig->finfo_[0];

///\todo should really copy over the data as well.
#ifdef DO_UNIT_TESTS
		numInstances++;
#endif	
		;
}

ArrayElement::~ArrayElement()
{
#ifndef DO_UNIT_TESTS
	// The unit tests create ArrayElement without any finfos.
	assert( finfo_.size() > 0 );
#endif	
#ifdef DO_UNIT_TESTS
	numInstances--;
#endif

	/**
	 * \todo Lots of cleanup stuff here to implement.
	// Find out what data is, and call its delete command.
	ThisFinfo* tf = dynamic_cast< ThisFinfo* >( finfo_[0] );
	tf->destroy( data() );
	*/	
	if ( data_ ) {
		if ( finfo_.size() > 0 && finfo_[0] != 0 ) {
			ThisFinfo* tf = dynamic_cast< ThisFinfo* >( finfo_[0] );
			if ( tf && tf->noDeleteFlag() == 0 )
				finfo_[0]->ftype()->destroy( data_, 1 );
		}
	}

	/**
	 * Need to explicitly drop messages, because we cannot tie the 
	 * operation to the Msg destructor. This is because the Msg vector
	 * changes size all the time but the Msgs themselves should not
	 * be removed.
	 * Note that we don't use DropAll, because by the time the call has
	 * come here we should have cleared out all the messages going outside
	 * the tree being deleted. Here we just destroy the allocated
	 * ConnTainers and their vectors in all messages.
	 */
	vector< Msg >::iterator m;
	for ( m = msg_.begin(); m!= msg_.end(); m++ )
		m->dropForDeletion();

	// Check if Finfo is one of the transient set, if so, clean it up.
	vector< Finfo* >::iterator i;
	// cout << name() << " " << id() << " f = ";
	for ( i = finfo_.begin(); i != finfo_.end(); i++ ) {
		assert( *i != 0 );
		// cout << ( *i )->name()  << " ptr= " << *i << " " ;
		if ( (*i)->isTransient() ) {
			delete *i;
		}
	}
	// cout << endl;
}

const std::string& ArrayElement::className( ) const
{
	return cinfo()->name();
}

const Cinfo* ArrayElement::cinfo( ) const
{
	const ThisFinfo* tf = dynamic_cast< const ThisFinfo* >( finfo_[0] );
	assert( tf != 0 );
	return tf->cinfo();
}

//////////////////////////////////////////////////////////////////
// Msg traversal functions
//////////////////////////////////////////////////////////////////

/**
 * The Conn iterators have to be deleted by the recipient function.
 */
Conn* ArrayElement::targets( int msgNum ) const
{
	if ( msgNum >= 0 && 
		static_cast< unsigned int >( msgNum ) < cinfo()->numSrc() )
		return new TraverseMsgConn( &msg_[ msgNum ], this, 0 );
	else if ( msgNum < 0 ) {
		const vector< ConnTainer* >* d = dest( msgNum );
		if ( d )
			return new TraverseDestConn( d, 0 );
	}
	return 0;
}

/**
 * The Conn iterators have to be deleted by the recipient function.
 */
Conn* ArrayElement::targets( const string& finfoName ) const
{
	const Finfo* f = cinfo()->findFinfo( finfoName );
	if ( !f )
		return 0;
	return targets( f->msg() );
}

unsigned int ArrayElement::numTargets( int msgNum ) const
{
	if ( msgNum >= 0 && 
		static_cast< unsigned int >( msgNum ) < cinfo()->numSrc() )
		return msg_[ msgNum ].numTargets( this );
	else if ( msgNum < 0 ) {
		const vector< ConnTainer* >* d = dest( msgNum );
		if ( d )
			return d->size();
	}
	return 0;
}

unsigned int ArrayElement::numTargets( const string& finfoName ) const
{
	const Finfo* f = cinfo()->findFinfo( finfoName );
	if ( !f )
		return 0;
	return numTargets( f->msg() );
}

//////////////////////////////////////////////////////////////////
// Msg functions
//////////////////////////////////////////////////////////////////

const Msg* ArrayElement::msg( unsigned int msgNum ) const
{
	assert ( msgNum < msg_.size() );
	return ( &( msg_[ msgNum ] ) );
}

Msg* ArrayElement::varMsg( unsigned int msgNum )
{
	assert ( msgNum < msg_.size() );
	return ( &( msg_[ msgNum ] ) );
}

const vector< ConnTainer* >* ArrayElement::dest( int msgNum ) const
{
	if ( msgNum >= 0 )
		return 0;
	map< int, vector< ConnTainer* > >::const_iterator i = dest_.find( msgNum );
	if ( i != dest_.end() ) {
		return &( *i ).second;
	}
	return 0;
}

vector< ConnTainer* >* ArrayElement::getDest( int msgNum ) 
{
	return &dest_[ msgNum ];
}

/*
const Msg* ArrayElement::msg( const string& fName )
{
	const Finfo* f = findFinfo( fName );
	if ( f ) {
		int msgNum = f->msg();
		if ( msgNum < msg_.size() )
			return ( &( msg_[ msgNum ] ) );
	}
	return 0;
}
*/

unsigned int ArrayElement::addNextMsg()
{
	msg_.push_back( Msg() );
	return msg_.size() - 1;
}

unsigned int ArrayElement::numMsg() const
{
	return msg_.size();
}

//////////////////////////////////////////////////////////////////
// Information functions
//////////////////////////////////////////////////////////////////

unsigned int ArrayElement::getTotalMem() const
{
	return sizeof( ArrayElement ) + 
		sizeof( name_ ) + name_.length() + 
		sizeof( finfo_ ) + finfo_.size() * sizeof( Finfo* ) +
		getMsgMem();
}

unsigned int ArrayElement::getMsgMem() const
{
	vector< Msg >::const_iterator i;
	unsigned int ret = 0;
	for ( i = msg_.begin(); i < msg_.end(); i++ ) {
		ret += i->size();
	}
	return ret;
}

bool ArrayElement::isMarkedForDeletion() const
{
	if ( finfo_.size() > 0 )
		return finfo_.back() == DeletionMarkerFinfo::global();
	// This fallback case should only occur during unit testing.
	return 0;
}

bool ArrayElement::isGlobal() const
{
	if ( finfo_.size() > 0 )
		return finfo_.back() == GlobalMarkerFinfo::global();
	// This fallback case should only occur during unit testing.
	return 0;
}


//////////////////////////////////////////////////////////////////
// Finfo functions
//////////////////////////////////////////////////////////////////

/**
 * Returns a finfo matching the target name.
 * Note that this is not a const function because the 'match'
 * function may generate dynamic finfos on the fly. If you need
 * a simpler, const string comparison then use constFindFinfo below,
 * which has limitations for special fields and arrays.
 */
const Finfo* ArrayElement::findFinfo( const string& name )
{
	vector< Finfo* >::reverse_iterator i;
	const Finfo* ret;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			ret = (*i)->match( this, name );
			if ( ret )
					return ret;
	}
	return 0;
}

/**
 * This is a const version of findFinfo. Instead of match it does a
 * simple strcmp against the field name. Cannot handle complex fields
 * like ones with indices.
 */
const Finfo* ArrayElement::constFindFinfo( const string& name ) const
{
	vector< Finfo* >::const_reverse_iterator i;
	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			if ( (*i)->name() == name )
				return *i;
	}

	// If it is not on the dynamically created finfos, maybe it is on
	// the static set.
	return cinfo()->findFinfo( name );
	
	return 0;
}

const Finfo* ArrayElement::findFinfo( const ConnTainer* c ) const
{
	vector< Finfo* >::const_reverse_iterator i;
	const Finfo* ret;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	// Reverse iterate because the zeroth finfo is the base,
	// and we want more recent finfos to override old ones.
	for ( i = finfo_.rbegin(); i != finfo_.rend(); i++ )
	{
			ret = (*i)->match( this, c );
			if ( ret )
					return ret;
	}
	return 0;
}

const Finfo* ArrayElement::findFinfo( int msgNum ) const
{
	const Cinfo* c = cinfo();
	return c->findFinfo( msgNum );
}

const Finfo* ArrayElement::localFinfo( unsigned int index ) const
{
	if ( index >= finfo_.size() ) 
		return 0;
	return finfo_[ index ];
}

unsigned int ArrayElement::listFinfos( 
				vector< const Finfo* >& flist ) const
{
	vector< Finfo* >::const_iterator i;

	// We should always have a base finfo.
	assert( finfo_.size() > 0 );

	for ( i = finfo_.begin(); i != finfo_.end(); i++ )
	{
		(*i)->listFinfos( flist );
	}

	return flist.size();
}

unsigned int ArrayElement::listLocalFinfos( vector< Finfo* >& flist )
{
	flist.resize( 0 );
	if ( finfo_.size() <= 1 )
		return 0;
	flist.insert( flist.end(), finfo_.begin() + 1, finfo_.end() );
	return flist.size();
}

void ArrayElement::addExtFinfo(Finfo *f){
	//don't think anything just add the finfo to the list
	finfo_.push_back(f);
}

/**
 * Here we need to put in the new Finfo, and also check if it
 * requires allocation of any MsgSrc or MsgDest slots.
 */
void ArrayElement::addFinfo( Finfo* f )
{
	unsigned int num = msg_.size();
	f->countMessages( num );
	if ( num > msg_.size() )
		msg_.resize( num );
	finfo_.push_back( f );
}

/**
 * This function cleans up the finfo f. It removes its messages,
 * deletes it, and removes its entry from the finfo list. Returns
 * true if the finfo was found and removed. At this stage it does NOT
 * permit deleting the ThisFinfo at index 0.
 */
bool ArrayElement::dropFinfo( const Finfo* f )
{
	if ( finfo_.size() < 2 )
		return 0;

	vector< Finfo* >::iterator i;
	for ( i = finfo_.begin() + 1; i != finfo_.end(); i++ ) {
		if ( *i == f ) {
			assert ( f->msg() < static_cast< int >( msg_.size() ) );
			msg_[ f->msg() ].dropAll( this );
			delete *i;
			finfo_.erase( i );
			return 1;
		}
	}
	return 0;
}

void ArrayElement::setThisFinfo( Finfo* f )
{
	if ( finfo_.size() == 0 )
		finfo_.resize( 1 );
	finfo_[0] = f;
}

const Finfo* ArrayElement::getThisFinfo( ) const
{
	if ( finfo_.size() == 0 )
		return 0;
	return finfo_[0];
}


void ArrayElement::prepareForDeletion( bool stage )
{
	if ( stage == 0 ) {
		finfo_.push_back( DeletionMarkerFinfo::global() );
	} else { // Delete all the remote conns that have not been marked.
		vector< Msg >::iterator m;
		for ( m = msg_.begin(); m!= msg_.end(); m++ ) {
			m->dropRemote();
		}

		// Delete the dest connections too
		map< int, vector< ConnTainer* > >::iterator j;
		for ( j = dest_.begin(); j != dest_.end(); j++ ) {
			Msg::dropDestRemote( j->second );
		}
	}
}

/**
 * Debugging function to print out msging info
 */
void ArrayElement::dumpMsgInfo() const
{
	unsigned int i;
	cout << "Element " << name_ << ":\n";
	cout << "msg_: funcid, sizes\n";
	for ( i = 0; i < msg_.size(); i++ ) {
		vector< ConnTainer* >::const_iterator j;
		cout << i << ":	funcid =" << msg_[i].funcId() << ": ";
		for ( j = msg_[i].begin(); j != msg_[i].end(); j++ )
			cout << ( *j )->size() << ", ";
	}
	cout << endl;
}

// Overrides Element version.
Id ArrayElement::id() const {
	return Element::id().assignIndex( Id::AnyIndex );
}

#ifdef DO_UNIT_TESTS

/**
 * Here we define a test class that sends 'output' to 'input' at
 * 'process'.
 * It does not do any numerics at all.
 */

Slot outputSlot;

class Atest {
	public: 
		static void setInput( const Conn* c, double value ) {
			static_cast< Atest* >( c->data() )->input_ = value;
		}
		static double getInput( Eref e ) {
			return static_cast< Atest* >( e.data() )->input_;
		}
		static void setOutput( const Conn* c, double value ) {
			static_cast< Atest* >( c->data() )->output_ = value;
		}
		static double getOutput( Eref e ) {
			return static_cast< Atest* >( e.data() )->output_;
		}
		static void process( Eref e ) {
			send1< double >( e, outputSlot, getOutput( e ) );
		}
	private:
		double input_;
		double output_;
};

const Cinfo* initAtestCinfo()
{
	static Finfo* aTestFinfos[] = 
	{
		new ValueFinfo( "input", ValueFtype1< double >::global(),
			GFCAST( &Atest::getInput ),
			RFCAST( &Atest::setInput )
		),
	
		new ValueFinfo( "output", ValueFtype1< double >::global(),
			GFCAST( &Atest::getOutput ),
			RFCAST( &Atest::setOutput )
		),
		new SrcFinfo( "outputSrc", Ftype1< double >::global() ),
		new DestFinfo( "msgInput", Ftype1< double >::global(),
			RFCAST( &Atest::setInput )
		),
	};

	static Cinfo aTest( "Atest", "Upi", "Array Test class",
		initNeutralCinfo(),
		aTestFinfos,
		sizeof( aTestFinfos ) / sizeof( Finfo* ),
		ValueFtype1< Atest >::global()
	);

	return &aTest;
}

/**
 * This tests message passing within an ArrayElement, from one entry
 * to the next. One can force the connOption to a specific value,
 * which works for Simple and Many2Many. Should also work for
 * One2Many and Many2One.
 */
static const unsigned int NUMKIDS = 12;
Element* arrayElementInternalTest( unsigned int connOption )
{
	cout << "\nTesting Array Elements, option= " << connOption << ": ";

	const Cinfo* aTestCinfo = initAtestCinfo();

	FuncVec::sortFuncVec();
	outputSlot = aTestCinfo->getSlot( "outputSrc" );

	Element* n = Neutral::create( "Neutral", "n", 
		Element::root(), Id::scratchId() ); 

	Id childId = Id::scratchId();
	Element* child = 
		Neutral::createArray( "Atest", "foo", n, childId, NUMKIDS );

	ASSERT( child != 0, "Array Element" );
	ASSERT( child == childId(), "Array Element" );
	ASSERT( childId.index() == 0, "Array Element" );
	ASSERT( child->id().index() == Id::AnyIndex, "Array Element" );

	vector< Id > kids;
	bool ret = get< vector< Id > >( n, "childList", kids );
	ASSERT( ret, "Array kids" );
	ASSERT( kids.size() == NUMKIDS, "Array kids" );
	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		ASSERT( kids[i].index() == i, "Array kids" );
		int index;
		bool ret = get< int >( kids[i].eref(), "index", index );
		ASSERT( ret && index == static_cast< int >( i ), "Array kids" );
		double output = i;
		bool sret = set< double >( kids[i].eref(), "output", output );
		output = 0.0;
		ASSERT( sret, "Array kids" );
		sret = set< double >( kids[i].eref(), "input", 0.0 );
		ret = get< double >( kids[i].eref(), "output", output );
		ASSERT( sret && ret && ( output == i ), "Array kid assignment" );
	}

	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		if ( i > 0 ) {
			ret = kids[i-1].eref().add( "outputSrc",
				kids[i].eref(), "msgInput", connOption );
			ASSERT( ret, "Array msg setup" );
		}
	}
	for ( unsigned int i = 0 ; i < NUMKIDS - 1; i++ ) {
		double output = i * i + 1.0;
		bool ret = set< double >( kids[i].eref(), "output", output );
		Atest::process( kids[i].eref() );
		if ( i > 0 ) {
			double input = 0.0;
			double result = ( i - 1 ) * ( i - 1 ) + 1.0;
			ret = get< double >( kids[i].eref(), "input", input );
			ASSERT( ret && ( input == result ), "Array kid messaging" );
		}
	}
	return n;
}

void arrayElementTest()
{
	Element* n = arrayElementInternalTest( ConnTainer::Simple ); 
	set( n, "destroy" );
	n = arrayElementInternalTest( ConnTainer::Many2Many ); 

	Element* m = Neutral::create( "Neutral", "m", Element::root(), Id::scratchId() ); 
	Id destId = Id::scratchId();
	Element* dest = 
		Neutral::createArray( "Atest", "dest", m, destId, NUMKIDS );

	ASSERT( dest != 0, "Array Element" );
	ASSERT( dest == destId(), "Array Element" );
	ASSERT( destId.index() == 0, "Array Element" );
	ASSERT( dest->id().index() == Id::AnyIndex, "Array Element" );

	vector< Id > destKids;
	bool ret = get< vector< Id > >( m, "childList", destKids );
	ASSERT( ret, "Array kids" );
	ASSERT( destKids.size() == NUMKIDS, "Array kids" );

	// ret = childId.eref().add( "axial", destId.eref(), "raxial" );
	ASSERT( ret, "Array Many2Many msgs" );

	for ( unsigned int i = 0 ; i < NUMKIDS; i++ ) {
		for ( unsigned int j = 0 ; j < NUMKIDS; j++ ) {
			if ( i + j == NUMKIDS || i - j == NUMKIDS ) 
				continue;
		/*
		double Vm = 0;
		bool ret = get< double >( kids[i].eref(), "Vm", Vm );
		ASSERT( ret && Vm == i, "Array kids" );
		*/
			// ret = kids[i].eref().add( "axial", destKids[j].eref(), "raxial" );
			ASSERT( ret, "Array Many2Many msgs" );
		}
	}

	set( n, "destroy" );
	set( m, "destroy" );
}
#endif
