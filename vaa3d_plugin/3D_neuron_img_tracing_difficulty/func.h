#ifndef FUNC_H
#define FUNC_H

#include "reliable_detection_plugin.h"
#include "sort_swc.h"

struct input_PARA
{
    QString inimg_file,autoswc_file,manualswc_file;
    QString outimg_good_block,outimg_bad_block,outautoswc_good_block,outautoswc_bad_block,outmanualswc_good_block,outmanualswc_bad_block;
    QString neuron_overlap_name;
    V3DLONG channel;
};

struct overlap_neuron
{
    QString manual_file_name,auto_file_name;
    V3DLONG block_num;
};

QStringList importFileList_addnumbersort(const QString & curFilePath);
void get_blocks(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent);
void distance(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent);
bool export_TXT(vector<V3DLONG> &vec,QString fileSaveName);
void detective_neuron_num_type(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent);
bool neuron_num_TXT(vector<V3DLONG> &vec,vector<QString> &name,QString fileSaveName);
bool delete_few_tree(const V3DPluginArgList &input,V3DPluginArgList &output);
void test(const V3DPluginArgList &input,V3DPluginArgList &output);
bool different_confidence_area_samples_method1(V3DPluginCallback2 &callback,input_PARA &P);
bool different_confidence_area_samples_OneToOne(V3DPluginCallback2 &callback,input_PARA &P);
bool neuron_tree_num_TXT(vector<int> &vec1,vector<int> &vec2,vector<QString> &name,QString fileSaveName);
void detective_neuron_tree_num(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent);
bool delete_rag_branch_and_few_seg(const V3DPluginArgList &input,V3DPluginArgList &output);
bool delete_rag_branch(const V3DPluginArgList &input,V3DPluginArgList &output);
bool delete_few_seg(const V3DPluginArgList &input,V3DPluginArgList &output);
bool compute_blocks_distance(const V3DPluginArgList &input,V3DPluginArgList &output);

#endif // FUNC_H
