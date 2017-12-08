function r = tryCompileSPAMS()
cd spams-matlab
compile();
cd ..
r = testSPAMS();    
return
