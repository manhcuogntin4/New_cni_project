import os
directory="./"
def get_list_subdirectory(directory):
	x=[x[0] for x in os.walk(directory)]
	return x
x=get_list_subdirectory(directory)
print x[1:]