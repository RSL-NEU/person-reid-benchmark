#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define PARENT(a) (floor((a-1)/2))
#define FIRSTCHILD(a) (a*2+1)
#define SECONDCHILD(a) (a*2+2)
#define SQR(X) (X)*(X)

#if NAN_EQUALS_ZERO
#define IsNonZero(d) ((d) != 0.0 || mxIsNaN(d))
#else
#define IsNonZero(d) ((d) != 0.0)
#endif




typedef class heaptree  {

public:
double **heapnodes;
int *heapsize,**heapdata;
unsigned heapmaxsize;

heaptree();
void initheaptree(unsigned int n, unsigned int d);
void printheap(int heapindex);
void heapswaproot(int heapindex,double key,int data);
void heapupdate(int heapindex,double key,int data);
void heappoproot(int heapindex);
void cleanup(unsigned int n);

protected:
void heapswap(int heapindex, int ind1, int ind2);
void heapinsert(int heapindex,double key,int data);
inline double distance(double *v1, double *v2, int n);
inline double dotp(double *v1, double *v2, int n);

} heaptree_t;




inline double heaptree::distance(double *v1, double *v2, int n) {
 int i;
 double dist=0;

 for(i=0;i<n;i++) dist+=SQR(v1[i]-v2[i]);
 return(dist);
}



void heaptree::printheap(int heapindex){
 int i;
 printf("Index %i (%i): ", heapindex,heapsize[heapindex]);
 for(i=0;i<heapsize[heapindex];i++) printf(" %i %f ",heapdata[heapindex][i],heapnodes[heapindex][i]);
printf("\n");
}


void heaptree::heapswap(int heapindex, int ind1, int ind2){
 double tempkey;
 int tempdata;

  tempkey=heapnodes[heapindex][ind1];
  heapnodes[heapindex][ind1]=heapnodes[heapindex][ind2];
  heapnodes[heapindex][ind2]=tempkey;

  tempdata=heapdata[heapindex][ind1];
  heapdata[heapindex][ind1]=heapdata[heapindex][ind2];
  heapdata[heapindex][ind2]=tempdata;

/* printf("Swapped %i: %i<>%i\n",heapindex,ind1,ind2); */
}


void heaptree::heapinsert(int heapindex,double key,int data){
 int ind=heapsize[heapindex],pa;

/* printf("Insert %i: Value: %f Data:%i \n",heapindex,key,data);*/
 heapsize[heapindex]++; 
 heapnodes[heapindex][ind]=key;
 heapdata[heapindex][ind]=data;


 pa=(int) PARENT((double)ind);
 while(ind>0 && heapnodes[heapindex][pa]<heapnodes[heapindex][ind]){
  heapswap(heapindex,ind,pa);
  ind=pa;
  pa=(int) PARENT((double)ind);
 } 
};

void heaptree::heapswaproot(int heapindex,double key,int data){
 int ind=0,child1=0,child2=0,bigchild,hsize=heapsize[heapindex];
 double key1,key2,bigkey;

 heapnodes[heapindex][ind]=key;  /*overwrite key*/
 heapdata[heapindex][ind]=data;  /*overwrite data*/

 while(1){
  child1=FIRSTCHILD(ind);
  child2=SECONDCHILD(ind);  
  if(child2>=hsize) 
    {if(child1>=hsize) break; else {
       bigkey=heapnodes[heapindex][child1];
       bigchild=child1;}
    } else{
   key1=heapnodes[heapindex][child1]; 
   key2=heapnodes[heapindex][child2];  
   if(key1>key2) {bigkey=key1;bigchild=child1;} else {bigkey=key2;bigchild=child2;}
  }
  if(bigkey>key) heapswap(heapindex,ind,bigchild); else break;
  ind=bigchild;
 }
/* printheap(heapindex);*/
};

void heaptree::heapupdate(int heapindex,double key,int data){
 /* check if an element should be entered into the tree and if so, do so */
 if(heapsize[heapindex]<heapmaxsize) heapinsert(heapindex,key,data);
 else if(heapnodes[heapindex][0]>key) heapswaproot(heapindex,key,data);
}

void heaptree::heappoproot(int heapindex){
 /* remove the root of the tree and fix the remaining structure */

 /* if tree is of size 0, return */
 if(heapsize[heapindex]==0)	return;
 /* take last elemnt */
 double key=heapnodes[heapindex][heapsize[heapindex]-1];
 int data=heapdata[heapindex][heapsize[heapindex]-1];  
 heapsize[heapindex]--;	
 /* and overwrite the root*/
 heapswaproot(heapindex,key,data);
}


heaptree::heaptree(){};

void heaptree::initheaptree( unsigned int n, unsigned int d){
 int i;
  heapmaxsize=d;
//  heapsize=savemalloc(n*sizeof(int));
  heapsize=new  int [n];
 if(heapsize==NULL) mexErrMsgTxt("Out of Memory!\n");
// heapnodes=savemalloc(n*sizeof(double * ));
 heapnodes=new double * [n];
 if(heapnodes==NULL) mexErrMsgTxt("Out of Memory!\n");
// heapdata =savemalloc(n*sizeof(unsigned int * ));
 heapdata=new  int * [n];
 if(heapdata==NULL) mexErrMsgTxt("Out of Memory!\n");
 for(i=0;i<n;i++){ 
  //heapnodes[i]=savemalloc(heapmaxsize*sizeof(double));
heapnodes[i]=new double [heapmaxsize];
//  heapdata[i]=savemalloc(heapmaxsize*sizeof(int));
  heapdata[i]=new  int [heapmaxsize];
  heapsize[i]=0;
  if(heapnodes[i]==NULL) mexErrMsgTxt("Out of Memory!\n");
  if(heapdata[i]==NULL) mexErrMsgTxt("Out of Memory!\n");
 }
}


void heaptree::cleanup( unsigned int n){
 int i;

 for(i=0;i<n;i++){ 
	delete(heapnodes[i]);
    delete(heapdata[i]);
 }
 delete(heapnodes);
 delete(heapdata);
}

//   END OF HEAP TREE DATA STRUCTURE (should really be in its own file)
