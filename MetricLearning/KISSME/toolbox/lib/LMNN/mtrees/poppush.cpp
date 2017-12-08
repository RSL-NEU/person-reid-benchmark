

typedef class poppush  {

public:
int curpointer;
int stacksize;
int *nodes;
double *mindists;

poppush();
int pop(int *node,double*dist);
int push(int node,double dist);
void init(unsigned int n);
void cleanup();
void clear();
} poppush_t;


poppush::poppush(){}; // dummy initialization

int poppush::pop(int * node, double * dist){
	if(curpointer==0) return(-1);
	curpointer--;
	node[0]=nodes[curpointer];
	dist[0]=mindists[curpointer];
	return(curpointer);
}

void poppush::clear(){
	curpointer=0;
}

int poppush::push(int node, double dist){
	if(curpointer==stacksize) return(-1);
	nodes[curpointer]=node;
	mindists[curpointer]=dist;
	curpointer++;
	return(curpointer);
}


void poppush::init( unsigned int n){
	stacksize=n;
	curpointer=0;
	mindists=new  double [n];	
    if(mindists==NULL) mexErrMsgTxt("Out of Memory!\n");
	nodes=new  int [n];	
    if(nodes==NULL) mexErrMsgTxt("Out of Memory!\n");
}	

void poppush::cleanup(){
	delete(mindists);
	delete(nodes);	
}

//   END OF HEAP TREE DATA STRUCTURE (should really be in its own file)
