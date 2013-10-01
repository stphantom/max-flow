CC = gcc
CFLAGS = -pthread -O3

all: amf_d

amf_d: amf_d.c amf.h
	${CC} ${CFLAGS} -o amf_d amf_d.c

clean:
	rm -fr *.o amf_d
