#!/usr/bin/python
#coding:utf-8

import os

startTag = '<us-patent-grant lang="EN"'
endTag = '</us-patent-grant>'

SAVLIST = (('<application-reference', '</application-reference>'), ('<main-cpc>', '</main-cpc>'),
           ('<invention-title', '</invention-title>'), ('<assignee>', '</assignee>'),
           ('<abstract', '</abstract>'))

openFileHandle = open('Dataset/ipg150106.xml', 'r')

text = openFileHandle.readlines()


# for (start, end) in SAVLIST:
#     print start,end
for line in text:

    flag = False

    if startTag in line:
        fileName = line[63:73]
        writeFileHandle = open('Dataset/150106/' + fileName + '.xml', 'w')
        writeFileHandle.write(line)

    if SAVLIST[0][0] in line:
        for content in text:
            if id(content) == id(line):
                flag = True
            if flag:
                writeLine ='%s'%content

                writeFileHandle.write(writeLine)
            if SAVLIST[0][1] in content:
                #writeFileHandle.write(content)
                flag = False
                continue

    elif SAVLIST[1][0] in line:
        for content in text:
            if id(content) == id(line):
                flag = True
            if flag:
                writeLine ='%s'%content

                writeFileHandle.write(writeLine)
            if SAVLIST[1][1] in content:
                #writeFileHandle.write(content)
                flag = False
                continue

    elif SAVLIST[2][0] in line:
        for content in text:
            if id(content) == id(line):
                flag = True
            if flag:
                writeLine ='%s'%content
  
                writeFileHandle.write(writeLine)
            if SAVLIST[2][1] in content:
                #writeFileHandle.write(content)
                flag = False
                continue

    elif SAVLIST[3][0] in line:
        for content in text:
            if id(content) == id(line):
                flag = True
            if flag:
                writeLine ='%s'%content

                writeFileHandle.write(writeLine)
            if SAVLIST[3][1] in content:
                #writeFileHandle.write(content)
                flag = False
                continue

    elif SAVLIST[4][0] in line:
        for content in text:
            if id(content) == id(line):
                flag = True
            if flag:
                writeLine ='%s'%content

                writeFileHandle.write(writeLine)
            if SAVLIST[4][1] in content:
                #writeFileHandle.write(content)
                flag = False
                continue

    if endTag in line:
        writeFileHandle.write(line)
        print fileName + '.xml has been saved!'



openFileHandle.close()
writeFileHandle.close()
# os.remove('filename')
# os.rename('Temp','filename')
print "All Done!"
