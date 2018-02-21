import requests
import glob
import sys
import json
import io
import os.path
import subprocess
from pandas import DataFrame
#http://docs.python-requests.org/en/latest/user/quickstart/#post-a-multipart-encoded-file

def readFileImages(strFolderName):
	print strFolderName
	image_list = []
	st=strFolderName+"*.png"
	for filename in glob.glob(st): #assuming gif
	    image_list.append(filename)
	return image_list


def writeFileOCR(f, txt_ocr):
	f.write(txt_ocr)
	return annotations


def readFileID(file):
	#file = open('testfile.txt', 'r') 
	ls_ID=[]
	for line in file:
		ls_ID.append(line.rstrip())
	return ls_ID

file_ID= os.popen("ls -v  test/*.png")
ls=readFileID(file_ID)

#pathFolder="./test_cni/"
#ls=readFileImages(pathFolder)

print ls

url = "http://localhost:5000/ocr/"
url_type="http://localhost:5000/"


file = open('test_1000.txt','a') 
file_type=open('test_1000_type.txt','a') 
ls_nom, ls_epouse, ls_prenom, ls_lieu, ls_date, ls_mrz1, ls_mrz2=[],[],[],[],[], [], []
ls_type=[]
for path in ls:
	fin = open(path, 'rb')
	files = {'file': fin}

	try:
	  r_type=requests.post(url_type, files=files)
	  doc_type_json=r_type.json()
	  doc_type=doc_type_json.get(u'result')[0][0]
	  ls_type.append(doc_type)
	except:
		ls_type.append("not found")
	finally:
		fin.close()

for path in ls:
	fin = open(path, 'rb')
	files = {'file': fin}

	try:
	  # r_type=requests.post(url_type, files=files)
	  # doc_type=r_type.json()
	  # print doc_type.get(u'result')

	  r = requests.post(url, files=files)
	  #print type(r.json()
	  t=r.json()
	  result=t.get(u'result')
	  if len(result)>1:
	  	nom=t.get(u"result")[1].get(u'nom')
	  	#print nom


	  if len(t.get(u'result'))>1:
	  	if t.get(u"result")[1].get(u'nom'):
	  		nom=t.get(u"result")[1].get(u'nom')[0]
	  		print nom
	  		if nom !="":
	  			ls_nom.append(nom)
	  		else:
	  			ls_nom.append("")
	  	else:
	  		ls_nom.append("Not found")
	  	if t.get(u"result")[1].get(u'prenom'):
	  		prenom=t.get(u"result")[1].get(u'prenom')[0]
	  		print prenom
	  		if prenom !="":
	  			ls_prenom.append(prenom)
	  		else:
	  			ls_prenom.append("Not found")
	  	else:
	  		ls_prenom.append("Not found")
	  	if t.get(u"result")[1].get(u'nomepouse'):
	  		epouse=t.get(u"result")[1].get(u'nomepouse')[0]
	  		print "epouse:",epouse
	  		if epouse !="":
	  			ls_epouse.append(epouse)
	  		else:
	  			ls_epouse.append("")
	  	else:
	  		ls_epouse.append("Not found")

	  	if t.get(u"result")[1].get(u'lieu'):
	  		lieu=t.get(u"result")[1].get(u'lieu')[0]
	  		print "lieu:",lieu
	  		if lieu !="":
	  			ls_lieu.append(lieu)
	  		else:
	  			ls_lieu.append("")
	  	else:
	  		ls_lieu.append("Not found")

	  	if t.get(u"result")[1].get(u'datenaissance'):
	  		datenaissance=t.get(u"result")[1].get(u'datenaissance')[0]
	  		print "date:",datenaissance
	  		if datenaissance !="":
	  			ls_date.append(datenaissance)
	  		else:
	  			ls_date.append("Not found")
	  	else:
	  		ls_date.append("Not found")

	  	if t.get(u"result")[1].get(u'mrz1'):
	  		mrz1=t.get(u"result")[1].get(u'mrz1')[0]
	  		print "mrz1:",mrz1
	  		if mrz1 !="":
	  			ls_mrz1.append(mrz1)
	  		else:
	  			ls_mrz1.append("Not found")
	  	else:
	  		ls_mrz1.append("Not found")
	  	
	  	if t.get(u"result")[1].get(u'mrz2'):
	  		mrz2=t.get(u"result")[1].get(u'mrz2')[0]
	  		print "mrz2:",mrz2
	  		if mrz2 !="":
	  			ls_mrz2.append(mrz2)
	  		else:
	  			ls_mrz2.append("Not found")
	  	else:
	  		ls_mrz2.append("Not found")
	  #print path
	  #print r.text

	  # if 'nom' in r.text:
	  # 	ls_nom.append(r.text['nom'][0])
	  file.write(path)
	  with open('test_1000.txt', 'a') as outfile:  
	  	json.dump(r.text, outfile)
	  file.write('/n')
	  #print r.text
	except:
		ls_epouse.append("Not found")
		ls_nom.append("Not found")
		ls_prenom.append("Not found")
		ls_date.append("Not found")
		ls_lieu.append("Not found")
		ls_mrz1.append("Not found")
		ls_mrz2.append("Not found")
		pass
	finally:
		fin.close()
#print ls_nom, ls_prenom, ls_epouse, ls_type
# s1 = pd.Series(ls_nom, name='Nom')
# s2 = pd.Series(ls_prenom, name='Prenom')
# s3= pd.Series(ls_epouse, name="Epouse")
# s3=pd.Series(ls_date, )
df=DataFrame({'File ID':ls, 'Nom':ls_nom, 'Nom Epouse':ls_epouse, 'Prenom': ls_prenom, 'Lieu': ls_lieu, 'Date': ls_date, 'MRZ1': ls_mrz1, 'MRZ2':ls_mrz2, 'Type detect':ls_type})
#df=DataFrame({'File ID':ls, 'Date': ls_date})
df.to_excel('test_v1.xlsx', sheet_name='Match_ID_File', index=False)
# file.close()