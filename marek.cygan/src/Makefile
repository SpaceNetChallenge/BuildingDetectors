%: %.cpp
	clang++ -o $@ $< -std=c++11 -Wall -W -lm -lpthread -lX11 -O3 

debug: vis.cpp
	g++ -o $@ $< -std=c++11 -Wall -W -lm -lpthread -lX11 -g -ggdb
