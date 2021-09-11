
TEMPLATE	= lib
CONFIG	+= qt plugin warn_off
#CONFIG	+= x86_64
VAA3DPATH = /home/zhang/v3d_external
INCLUDEPATH	+= $$VAA3DPATH/v3d_main/basic_c_fun
INCLUDEPATH	+= $$VAA3DPATH/v3d_main/common_lib/include

HEADERS	+= reliable_detection_plugin.h \
    func.h \
    sort_swc.h
SOURCES	+= reliable_detection_plugin.cpp \
    func.cpp \
    sort_swc.cpp
SOURCES	+= $$VAA3DPATH/v3d_main/basic_c_fun/v3d_message.cpp
SOURCES	+= $$VAA3DPATH/v3d_main/basic_c_fun/basic_surf_objs.cpp

TARGET	= $$qtLibraryTarget(reliable_detection)
DESTDIR	= $$VAA3DPATH/bin/plugins/reliable_detection_all_neuron/

