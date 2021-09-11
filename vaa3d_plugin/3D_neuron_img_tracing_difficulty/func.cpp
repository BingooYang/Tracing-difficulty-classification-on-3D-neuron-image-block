#include "func.h"
#include "math.h"

#include "../../../released_plugins/v3d_plugins/resample_swc/resampling.h"

#ifndef VOID
#define VOID 1000000000
#endif

#define MHDIS(a,b) ( (fabs((a).x-(b).x)) + (fabs((a).y-(b).y)) + (fabs((a).z-(b).z)) )
#define NTDIS(a,b) pow(pow(a.x-b.x,2)+pow(a.y-b.y,2)+pow(a.z-b.z,2),0.5)

//bool neuron_tree_num_TXT(vector<int> &vec1,vector<int> &vec2,vector<QString> &name,QString fileSaveName);
bool neuron_overlap_TXT2(vector<overlap_neuron> &vec1,QString fileSaveName);
bool neuron_overlap_TXT(vector<V3DLONG> &vec1,vector<V3DLONG> &vec2,vector<V3DLONG> &vec3,vector<V3DLONG> &vec4,QString fileSaveName);
QStringList importFileList_addnumbersort(const QString & curFilePath);
bool export_list2file(QList<NeuronSWC> & lN, QString fileSaveName, QString fileOpenName);
void test(const V3DPluginArgList &input,V3DPluginArgList &output)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));

    cout<<"manual_swcfiles_name:"<<manual_swcfiles[0].toStdString()<<endl;
    cout<<"auto_swcfiles_name:"<<auto_swcfiles[0].toStdString()<<endl;
}
bool different_confidence_area_samples_method1(V3DPluginCallback2 &callback,input_PARA &P)
{
    //load and processing swc
    QStringList autoswclist = importFileList_addnumbersort(P.autoswc_file);
    QStringList manualswclist = importFileList_addnumbersort(P.manualswc_file);
    cout<<"autoswclist = "<<autoswclist.size()<<endl;
    cout<<"manualswclist = "<<manualswclist.size()<<endl;

    //load image
//    QString imagepath = P.inimg_file;

    //set block size
    double l_x = 32;
    double l_y = 32;
    double l_z = 16;

    double step1=1.0;
    double step2 = 100;

    QList<NeuronTree> All_auto_nt;
    for(int i = 2;i < autoswclist.size();i++)
    {
        NeuronTree auto_nt = readSWC_file(autoswclist[i]);
        NeuronTree auto_nt_resample = resample(auto_nt,step1);
//        cout<<"auto_nt_resample = "<<auto_nt_resample.listNeuron.size()<<endl;

        All_auto_nt.push_back(auto_nt_resample);
    }

    //保存两个神经元重构到一个block中的变量
    vector<V3DLONG> neuron_one_num;
    vector<V3DLONG> neuron_two_num;
    vector<V3DLONG> block_num;
    vector<V3DLONG> overlap_num;

    cout<<"start compute"<<endl;

    vector<overlap_neuron> overlap_data;
    //get samples
//    for(int i = 2;i <7;i++)
    for(int i = 2;i < manualswclist.size();i++)
    {
        cout<<"Processing "<<i-1<<"/"<<manualswclist.size()<<" done!"<<endl;
        cout<<"manualswclist : "<<manualswclist[i].toStdString()<<endl;
        NeuronTree manual_nt = readSWC_file(manualswclist[i]);
        NeuronTree manual_nt_resample = resample(manual_nt,step1);

        for(V3DLONG j =0;j < manual_nt_resample.listNeuron.size();j=j+step2)
        {
            NeuronSWC cur;
            cur.x = manual_nt_resample.listNeuron[j].x;
            cur.y = manual_nt_resample.listNeuron[j].y;
            cur.z = manual_nt_resample.listNeuron[j].z;

            V3DLONG xb = cur.x-l_x;
            V3DLONG xe = cur.x+l_x-1;
            V3DLONG yb = cur.y-l_y;
            V3DLONG ye = cur.y+l_y-1;
            V3DLONG zb = cur.z-l_z;
            V3DLONG ze = cur.z+l_z-1;

            V3DLONG im_cropped_sz[4];
            unsigned char * im_cropped = 0;
            V3DLONG pagesz;

            pagesz = (xe-xb+1)*(ye-yb+1)*(ze-zb+1);
            im_cropped_sz[0] = xe-xb+1;
            im_cropped_sz[1] = ye-yb+1;
            im_cropped_sz[2] = ze-zb+1;
            im_cropped_sz[3] = 1;

            try {im_cropped = new unsigned char [pagesz];}
            catch(...)  {v3d_msg("cannot allocate memory for image_mip."); return false;}

            QList<NeuronSWC> out_autoswc,out_manualswc;
            QList<NeuronSWC> out_autoswc_sort,out_manualswc_sort;

            //manual swc block
            for(V3DLONG l = 0;l < manual_nt_resample.listNeuron.size();l++)
            {
                NeuronSWC S;
                if(manual_nt_resample.listNeuron[l].x<xe&&manual_nt_resample.listNeuron[l].x>xb&&manual_nt_resample.listNeuron[l].y<ye&&manual_nt_resample.listNeuron[l].y>yb&&manual_nt_resample.listNeuron[l].z<ze&&manual_nt_resample.listNeuron[l].z>zb)
                {
                    S.x = manual_nt_resample.listNeuron[l].x-xb;
                    S.y = manual_nt_resample.listNeuron[l].y-yb;
                    S.z = manual_nt_resample.listNeuron[l].z-zb;
                    S.n = manual_nt_resample.listNeuron[l].n;
                    S.pn = manual_nt_resample.listNeuron[l].pn;
                    S.r = manual_nt_resample.listNeuron[l].r;
                    S.type = manual_nt_resample.listNeuron[l].type;

                    out_manualswc.push_back(S);
                }
            }

            //auto block
            int lens2=0;
            for(V3DLONG k = 0; k < All_auto_nt.size();k++)
            {
                V3DLONG tem_num = 0;

                for(V3DLONG l= 0;l < All_auto_nt[k].listNeuron.size();l++)
                {
                    NeuronSWC S;
                    if(All_auto_nt[k].listNeuron[l].x<xe&&All_auto_nt[k].listNeuron[l].x>xb&&All_auto_nt[k].listNeuron[l].y<ye&&All_auto_nt[k].listNeuron[l].y>yb&&All_auto_nt[k].listNeuron[l].z<ze&&All_auto_nt[k].listNeuron[l].z>zb)
                    {
                        S.x = All_auto_nt[k].listNeuron[l].x-xb;
                        S.y = All_auto_nt[k].listNeuron[l].y-yb;
                        S.z = All_auto_nt[k].listNeuron[l].z-zb;
                        S.n = All_auto_nt[k].listNeuron[l].n+lens2;
                        S.pn = All_auto_nt[k].listNeuron[l].pn+lens2;
                        S.r = All_auto_nt[k].listNeuron[l].r;
                        S.type = All_auto_nt[k].listNeuron[l].type;

                        out_autoswc.push_back(S);

                        tem_num++;
                    }
                }


                lens2=lens2+All_auto_nt[k].listNeuron.size();

            }         

            V3DLONG rootid = VOID;
            V3DLONG thres = 0;
            SortSWC(out_autoswc, out_autoswc_sort ,rootid, thres);
            SortSWC(out_manualswc, out_manualswc_sort ,rootid, thres);

            //outfile name
            QString outimg_bad_file,out_autoswc_bad_file,out_manualswc_bad_file,outimg_good_file,out_autoswc_good_file,out_manualswc_good_file;
            //转化为三位数，比如0转化为000，12转化为012
            QString m = QString::number(v3d_sint64((i-1)/100))+QString::number(v3d_sint64((i-1)/10%10))+QString::number(v3d_sint64((i-1)%10));
            QString n = QString::number(v3d_sint64(j/100/1000))+QString::number(v3d_sint64(j/100/100%10))+QString::number(v3d_sint64(j/100/10%10))+QString::number(v3d_sint64(j/100%10));

            outimg_bad_file = P.outimg_bad_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".tif";
            out_autoswc_bad_file = P.outautoswc_bad_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";
            out_manualswc_bad_file = P.outmanualswc_bad_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";
            outimg_good_file = P.outimg_good_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".tif";
            out_autoswc_good_file = P.outautoswc_good_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";
            out_manualswc_good_file = P.outmanualswc_good_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";

            //bad and good block class
            if(out_autoswc.size()<10)
            {
                //save bad swc block
                export_list2file(out_autoswc_sort,out_autoswc_bad_file,out_autoswc_bad_file);
                export_list2file(out_manualswc_sort,out_manualswc_bad_file,out_manualswc_bad_file);

//                im_cropped = callback.getSubVolumeTeraFly(imagepath.toStdString(),xb,xe+1,yb,ye+1,zb,ze+1);
//                simple_saveimage_wrapper(callback, outimg_bad_file.toStdString().c_str(),(unsigned char *)im_cropped,im_cropped_sz,1);

            }else
            {
              //save good swc block
                export_list2file(out_autoswc_sort,out_autoswc_good_file,out_autoswc_good_file);
                export_list2file(out_manualswc_sort,out_manualswc_good_file,out_manualswc_good_file);

//                im_cropped = callback.getSubVolumeTeraFly(imagepath.toStdString(),xb,xe+1,yb,ye+1,zb,ze+1);
//                simple_saveimage_wrapper(callback, outimg_good_file.toStdString().c_str(),(unsigned char *)im_cropped,im_cropped_sz,1);
            }

//            if(im_cropped) {delete []im_cropped; im_cropped = 0;}

        }

    }
    //保存两个神经元重构的结果在一个block中的情况，也就是两个神经元重构的结果很接近
//    QString neuron_overlap_name= P.neuron_overlap_name + "/" +"neuron_overlap_data.txt";
//    cout<<"save file"<<endl;
//    neuron_overlap_TXT2(overlap_data, neuron_overlap_name);
//    neuron_overlap_TXT(neuron_one_num,neuron_two_num,block_num,overlap_num,neuron_overlap_name);

    return true;
}

//在与金标准对应的重构神经元中搜索
bool different_confidence_area_samples_OneToOne(V3DPluginCallback2 &callback,input_PARA &P)
{
    //load and processing swc
    QStringList autoswclist = importFileList_addnumbersort(P.autoswc_file);
    QStringList manualswclist = importFileList_addnumbersort(P.manualswc_file);
    cout<<"autoswclist = "<<autoswclist.size()<<endl;
    cout<<"manualswclist = "<<manualswclist.size()<<endl;

    //load image
//    QString imagepath = P.inimg_file;

    //set block size
    double l_x = 32;
    double l_y = 32;
    double l_z = 16;

    double step1=1.0;
    double step2 = 100;

//    QList<NeuronTree> All_auto_nt;
//    for(int i = 2;i < autoswclist.size();i++)
//    {
//        NeuronTree auto_nt = readSWC_file(autoswclist[i]);
//        NeuronTree auto_nt_resample = resample(auto_nt,step1);
////        cout<<"auto_nt_resample = "<<auto_nt_resample.listNeuron.size()<<endl;

//        All_auto_nt.push_back(auto_nt_resample);
//    }


    //get samples
//    for(int i = 2;i <3;i++)
    for(int i = 2;i < manualswclist.size();i++)
    {
//        cout<<"Processing "<<i-1<<"/"<<manualswclist.size()<<" done!"<<endl;
//        cout<<"manualswclist : "<<manualswclist[i].toStdString()<<endl;
        NeuronTree manual_nt = readSWC_file(manualswclist[i]);
        NeuronTree manual_nt_resample = resample(manual_nt,step1);
        //找到与金标准名字对应的自动重构神经元
        NeuronTree auto_nt,auto_nt_resample;
        for(V3DLONG a_i=2;a_i<autoswclist.size();a_i++)
        {

            QStringList name1 = manualswclist[i].split('/');
            QStringList name2 = autoswclist[a_i].split('/');
            if(name1[name1.count()-1] == name2[name2.count()-1])
            {
                auto_nt = readSWC_file(autoswclist[a_i]);
                auto_nt_resample = resample(auto_nt,step1);
                break;
            }
        }


        for(V3DLONG j =0;j < manual_nt_resample.listNeuron.size();j=j+step2)
        {
            NeuronSWC cur;
            cur.x = manual_nt_resample.listNeuron[j].x;
            cur.y = manual_nt_resample.listNeuron[j].y;
            cur.z = manual_nt_resample.listNeuron[j].z;

            V3DLONG xb = cur.x-l_x;
            V3DLONG xe = cur.x+l_x-1;
            V3DLONG yb = cur.y-l_y;
            V3DLONG ye = cur.y+l_y-1;
            V3DLONG zb = cur.z-l_z;
            V3DLONG ze = cur.z+l_z-1;

            V3DLONG im_cropped_sz[4];
            unsigned char * im_cropped = 0;
            V3DLONG pagesz;

            pagesz = (xe-xb+1)*(ye-yb+1)*(ze-zb+1);
            im_cropped_sz[0] = xe-xb+1;
            im_cropped_sz[1] = ye-yb+1;
            im_cropped_sz[2] = ze-zb+1;
            im_cropped_sz[3] = 1;

            try {im_cropped = new unsigned char [pagesz];}
            catch(...)  {v3d_msg("cannot allocate memory for image_mip."); return false;}

            QList<NeuronSWC> out_autoswc,out_manualswc;
            QList<NeuronSWC> out_autoswc_sort,out_manualswc_sort;

            //manual swc block
            for(V3DLONG l = 0;l < manual_nt_resample.listNeuron.size();l++)
            {
                NeuronSWC S;
                if(manual_nt_resample.listNeuron[l].x<xe&&manual_nt_resample.listNeuron[l].x>xb&&manual_nt_resample.listNeuron[l].y<ye&&manual_nt_resample.listNeuron[l].y>yb&&manual_nt_resample.listNeuron[l].z<ze&&manual_nt_resample.listNeuron[l].z>zb)
                {
                    S.x = manual_nt_resample.listNeuron[l].x-xb;
                    S.y = manual_nt_resample.listNeuron[l].y-yb;
                    S.z = manual_nt_resample.listNeuron[l].z-zb;
                    S.n = manual_nt_resample.listNeuron[l].n;
                    S.pn = manual_nt_resample.listNeuron[l].pn;
                    S.r = manual_nt_resample.listNeuron[l].r;
                    S.type = manual_nt_resample.listNeuron[l].type;

                    out_manualswc.push_back(S);
                }
            }

            //auto block
            for(V3DLONG l = 0;l < auto_nt_resample.listNeuron.size();l++)
            {
                NeuronSWC S;
                if(auto_nt_resample.listNeuron[l].x<xe&&auto_nt_resample.listNeuron[l].x>xb&&auto_nt_resample.listNeuron[l].y<ye&&auto_nt_resample.listNeuron[l].y>yb&&auto_nt_resample.listNeuron[l].z<ze&&auto_nt_resample.listNeuron[l].z>zb)
                {
                    S.x = auto_nt_resample.listNeuron[l].x-xb;
                    S.y = auto_nt_resample.listNeuron[l].y-yb;
                    S.z = auto_nt_resample.listNeuron[l].z-zb;
                    S.n = auto_nt_resample.listNeuron[l].n;
                    S.pn = auto_nt_resample.listNeuron[l].pn;
                    S.r = auto_nt_resample.listNeuron[l].r;
                    S.type = auto_nt_resample.listNeuron[l].type;

                    out_autoswc.push_back(S);
                }
            }

            V3DLONG rootid = VOID;
            V3DLONG thres = 0;
            SortSWC(out_autoswc, out_autoswc_sort ,rootid, thres);
            SortSWC(out_manualswc, out_manualswc_sort ,rootid, thres);

            //outfile name
            QString outimg_bad_file,out_autoswc_bad_file,out_manualswc_bad_file,outimg_good_file,out_autoswc_good_file,out_manualswc_good_file;
            //转化为三位数，比如0转化为000，12转化为012
            QString m = QString::number(v3d_sint64((i-1)/100))+QString::number(v3d_sint64((i-1)/10%10))+QString::number(v3d_sint64((i-1)%10));
            QString n = QString::number(v3d_sint64(j/100/1000))+QString::number(v3d_sint64(j/100/100%10))+QString::number(v3d_sint64(j/100/10%10))+QString::number(v3d_sint64(j/100%10));

            outimg_bad_file = P.outimg_bad_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".tif";
            out_autoswc_bad_file = P.outautoswc_bad_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";
            out_manualswc_bad_file = P.outmanualswc_bad_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";
            outimg_good_file = P.outimg_good_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".tif";
            out_autoswc_good_file = P.outautoswc_good_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";
            out_manualswc_good_file = P.outmanualswc_good_block+"/"+m+"_"+n+"_"+QString("x_%1_y_%2_z_%3").arg(xb).arg(yb).arg(zb)+".swc";

            //bad and good block class
            if(out_autoswc.size()<10)
            {
                //save bad swc block
                export_list2file(out_autoswc_sort,out_autoswc_bad_file,out_autoswc_bad_file);
                export_list2file(out_manualswc_sort,out_manualswc_bad_file,out_manualswc_bad_file);

//                im_cropped = callback.getSubVolumeTeraFly(imagepath.toStdString(),xb,xe+1,yb,ye+1,zb,ze+1);
//                simple_saveimage_wrapper(callback, outimg_bad_file.toStdString().c_str(),(unsigned char *)im_cropped,im_cropped_sz,1);

            }else
            {
              //save good swc block
                export_list2file(out_autoswc_sort,out_autoswc_good_file,out_autoswc_good_file);
                export_list2file(out_manualswc_sort,out_manualswc_good_file,out_manualswc_good_file);

//                im_cropped = callback.getSubVolumeTeraFly(imagepath.toStdString(),xb,xe+1,yb,ye+1,zb,ze+1);
//                simple_saveimage_wrapper(callback, outimg_good_file.toStdString().c_str(),(unsigned char *)im_cropped,im_cropped_sz,1);
            }

            //if(im_cropped) {delete []im_cropped; im_cropped = 0;}

        }

    }
    return true;
}
void detective_neuron_tree_num(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));

    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles,OneToOne_auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));

    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

    //output txt
    QString out_txt = QString(outlist->at(0))+"/"+"neuron_tree_num_816.txt";

    vector<V3DLONG> label;
    vector<QString> name;
    vector<int> manual_tree_num,auto_tree_num;

    for(int i = 0;i < manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));

        QStringList name1 = manual_swcfiles[i].split('/');
        name.push_back(name1[name1.count()-1]);


        int auto_trees_flag=0, manual_trees_flag=0;

        for (V3DLONG j=0; j<nt_Manual.listNeuron.size();j++)
        {
            //统计有几棵树
            if(nt_Manual.listNeuron[j].pn<0)
            {
                manual_trees_flag++;
            }

        }


        for (V3DLONG j=0; j<nt_Auto.listNeuron.size();j++)
        {
            //统计有几棵树
            if(nt_Auto.listNeuron[j].pn<0)
            {
                auto_trees_flag++;
            }

        }

        manual_tree_num.push_back(manual_trees_flag);
        auto_tree_num.push_back(auto_trees_flag);

    }
    cout<<"start save txt..."<<endl;
    neuron_tree_num_TXT(manual_tree_num,auto_tree_num,name,out_txt);
}

void detective_neuron_num_type(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    QStringList OneToOne_auto_swclist = importFileList_addnumbersort(QString(inlist->at(2)));

    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles,OneToOne_auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    for(int i = 2;i < OneToOne_auto_swclist.size();i++)
        OneToOne_auto_swcfiles.push_back(OneToOne_auto_swclist.at(i));

    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

    //output txt
    QString out_txt = QString(outlist->at(0))+"/"+"neuron_num.txt";

    vector<V3DLONG> label;
    vector<QString> name;
    for(int i = 0;i < manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));

        QStringList name1 = manual_swcfiles[i].split('/');
        name.push_back(name1[name1.count()-1]);

        V3DLONG OneToOne_auto_trees_flag=0,auto_trees_flag=0, manual_trees_flag=0;

        for (V3DLONG j=0; j<nt_Manual.listNeuron.size();j++)
        {
            //统计有几棵树
            if(nt_Manual.listNeuron[j].pn<0)
            {
                manual_trees_flag++;
            }

        }



        for(V3DLONG m=0;m<auto_swcfiles.size();m++)
        {
            QStringList name2 = auto_swcfiles[m].split('/');
            //找到与金标准神经元对应的自动重构block
            if(name1[name1.count()-1] == name2[name2.count()-1])
            {
                NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(m));

                for (V3DLONG j=0; j<nt_Auto.listNeuron.size();j++)
                {
                    if(nt_Auto.listNeuron[j].pn<0)
                    {
                        auto_trees_flag++;
                    }

                }
                break;
            }
        }

        for(V3DLONG m=0;m<OneToOne_auto_swcfiles.size();m++)
        {
            QStringList name2 = OneToOne_auto_swcfiles[m].split('/');
            //找到与金标准神经元对应的自动重构block
            if(name1[name1.count()-1] == name2[name2.count()-1])
            {
                NeuronTree nt_Auto_OneToOne = readSWC_file(OneToOne_auto_swcfiles.at(m));

                for (V3DLONG j=0; j<nt_Auto_OneToOne.listNeuron.size();j++)
                {
                    if(nt_Auto_OneToOne.listNeuron[j].pn<0)
                    {
                        OneToOne_auto_trees_flag++;
                    }

                }
                break;
            }
        }

        //给每个block打标签，auto和manual都只有一棵树则为0 ...
        if(OneToOne_auto_trees_flag == 0)
        {
            label.push_back(4);
        }
        else if(auto_trees_flag ==1 && manual_trees_flag ==1)
        {
            label.push_back(0);
        }else if(auto_trees_flag >1 && manual_trees_flag == 1)
        {
            label.push_back(1);
        }else if(auto_trees_flag ==1 && manual_trees_flag > 1)
        {
            label.push_back(2);
        }else if(auto_trees_flag >1 && manual_trees_flag >1)
        {
            label.push_back(3);
        }else
        {
            label.push_back(-1);
        }

    }
    cout<<"start save txt..."<<endl;
    neuron_num_TXT(label,name,out_txt);

}

bool save_blocks_distance_TXT(vector<V3DLONG> &vec,vector<QString> &name,QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<"num"<<"\t"<<"neuron_name"<<"blocks_distance"<<endl;
    for (int i=0;i<vec.size(); i++)
    {
        myfile <<i<<"\t"<<name.at(i)<<"\t"<< vec.at(i)<<"\t"<<endl;
    }

    file.close();
    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec.size()<<endl;
    return true;
}

bool compute_blocks_distance(const V3DPluginArgList &input,V3DPluginArgList &output)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manualswclist = importFileList_addnumbersort(QString(inlist->at(0)));
//    QStringList autoswclist = importFileList_addnumbersort(QString(inlist->at(1)));

//    cout<<"autoswclist = "<<autoswclist.size()<<endl;
    cout<<"manualswclist = "<<manualswclist.size()<<endl;

    //set block size
    double l_x = 32;
    double l_y = 32;
    double l_z = 16;

    double step1=1.0;
    double step2 = 100;

    vector<V3DLONG> dis_manual_block,dis_auto_block;
    vector<QString> manual_neuron_name;
    cout<<"start compute"<<endl;
//    for(int i = 2;i <7;i++)
    for(int i = 2;i < manualswclist.size();i++)
    {
        cout<<"Processing "<<i-1<<"/"<<manualswclist.size()<<" done!"<<endl;
        cout<<"manualswclist : "<<manualswclist[i].toStdString()<<endl;

        NeuronTree manual_nt = readSWC_file(manualswclist[i]);
        NeuronTree manual_nt_resample = resample(manual_nt,step1);

        QStringList name1 = manualswclist[i].split('/');
        QString name2 = name1[name1.count()-1];
        NeuronSWC pre_cor,current_cor;
        pre_cor.x = manual_nt_resample.listNeuron[0].x;
        pre_cor.y = manual_nt_resample.listNeuron[0].y;
        pre_cor.z = manual_nt_resample.listNeuron[0].z;

        for(V3DLONG j =1;j < manual_nt_resample.listNeuron.size();j=j+step2)
        {
            current_cor.x = manual_nt_resample.listNeuron[j].x;
            current_cor.y = manual_nt_resample.listNeuron[j].y;
            current_cor.z = manual_nt_resample.listNeuron[j].z;

            //添加距离
            dis_manual_block.push_back(NTDIS(current_cor,pre_cor));

            pre_cor.x = current_cor.x;
            pre_cor.y = current_cor.y;
            pre_cor.z = current_cor.z;

            QString m = QString::number(v3d_sint64((i-1)/100))+QString::number(v3d_sint64((i-1)/10%10))+QString::number(v3d_sint64((i-1)%10));
            QString n = QString::number(v3d_sint64((j-1)/100/1000))+QString::number(v3d_sint64((j-1)/100/100%10))+QString::number(v3d_sint64((j-1)/100/10%10))+QString::number(v3d_sint64((j-1)/100%10));

            manual_neuron_name.push_back(m+"_"+n+"_"+name2);
        }
    }
    QString blocks_distance_path = QString(outlist->at(0))+"blocks_distance.txt";
    save_blocks_distance_TXT(dis_manual_block,manual_neuron_name,blocks_distance_path);

}

bool save_nofrag_nofew_TXT(vector<QString> &vec,QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<"#####i neuron_name"<<endl;
    V3DLONG  p_pt=0;
    for (int i=0;i<vec.size(); i++)
    {
        //then save
        //myfile << p_pt->x<<"       "<<p_pt->y<<"       "<<p_pt->z<<"       "<<p_pt->signal<<endl;
        myfile <<i<<"\t"<< vec.at(i)<<endl;
    }

    file.close();
    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec.size()<<endl;
    return true;
}

bool detective_one_bound(V3DLONG x,V3DLONG y,V3DLONG z, V3DLONG t_xl,V3DLONG t_yl,V3DLONG t_zl,V3DLONG t_xr,V3DLONG t_yr,V3DLONG t_zr)
{
    if((x>=t_xl && x<=t_xr) && (y>=t_yl && y<=t_yr) && (z>=t_zl && z<=t_zr))
    {
        return 1;
    }else
    {
        return 0;
    }
}
bool detective_six_bound(V3DLONG x,V3DLONG y,V3DLONG z, V3DLONG t_xl,V3DLONG t_yl,V3DLONG t_zl,V3DLONG t_xr,V3DLONG t_yr,V3DLONG t_zr,int wide)
{
    cout<<"detective_six_bound"<<endl;
    if(detective_one_bound(x,y,z,t_xl,t_yl,t_zl,t_xl+wide,t_yl,t_zl)||
            detective_one_bound(x,y,z,t_xl,t_yl,t_zl,t_xl,t_yl+wide,t_zl)||
            detective_one_bound(x,y,z,t_xl,t_yl,t_zl,t_xl,t_yl,t_zl+wide)||
            detective_one_bound(x,y,z,t_xr,t_yr,t_zr,t_xr+wide,t_yr,t_zr)||
            detective_one_bound(x,y,z,t_xr,t_yr,t_zr,t_xr,t_yr+wide,t_zr)||
            detective_one_bound(x,y,z,t_xr,t_yr,t_zr,t_xr,t_yr,t_zr+wide))
    {
        //如果节点在边缘里返回0
        return 0;
    }else
    {
        return 1;
    }
}

//去掉毛刺(节点个数很少的分支)
bool delete_rag_branch(const V3DPluginArgList &input,V3DPluginArgList &output)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

//    //output file
//    QString out_manual_swc = QString(outlist->at(0));
//    QString out_auto_swc = QString(outlist->at(1));
    vector<QString> few_seg_manual_txt,few_seg_auto_txt;

//    for(int i = 0;i < 70;i++)
    for(int i = 0;i < manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));

/***********************************************************************************
 去毛刺：删掉分叉结点个数小于thred的毛刺
 **********************************************************************************/
        cout<<"start delete rag"<<endl;
        //删除manual毛刺
        QList<NeuronSWC> out_autoswc_no_rag_end,out_manualswc_no_rag_end;
        V3DLONG rag_num_manual=0,rag_num_auto=0,rag_record_manual=0,rag_record_auto=1,
                rag_delete_num_manual=0,rag_delete_num_auto=0;
        int thred_rag = 3;
        for (V3DLONG j=0; j<nt_Manual.listNeuron.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = nt_Manual.listNeuron[j].x;
            S.y = nt_Manual.listNeuron[j].y;
            S.z = nt_Manual.listNeuron[j].z;
            S.n = nt_Manual.listNeuron[j].n;
            S.pn = nt_Manual.listNeuron[j].pn;
            S.r = nt_Manual.listNeuron[j].r;
            S.type = nt_Manual.listNeuron[j].type;

            out_manualswc_no_rag_end.push_back(S);
            rag_num_manual++;

            if( j!=0 && (S.pn != (S.n-1)|| j == nt_Manual.listNeuron.size()-1))
            {
//                cout<<"S.n:"<<S.n<<' '<<S.pn<<endl;
                if(j != nt_Manual.listNeuron.size()-1)
                {
                    rag_num_manual--;
                    if(rag_num_manual <= thred_rag)
                    {
                        cout<<"delete1 rag"<<endl;
                        cout<<j<<' '<<rag_record_manual<<' '<<rag_num_manual<<endl;
                        //删除小于阈值的毛刺
                        out_manualswc_no_rag_end.erase(out_manualswc_no_rag_end.begin()+rag_record_manual-rag_delete_num_manual,out_manualswc_no_rag_end.begin()+j-rag_delete_num_manual);
                        rag_delete_num_manual += rag_num_manual;
                    }
                        rag_record_manual = j;
                }else
                {
                    if(rag_num_manual <= thred_rag)
                    {
                        cout<<"delete2 rag"<<endl;
                        cout<<j<<' '<<rag_record_manual<<' '<<rag_num_manual<<' '<<nt_Manual.listNeuron.size()<<endl;
                        //删除小于阈值的毛刺
                        out_manualswc_no_rag_end.erase(out_manualswc_no_rag_end.begin()+rag_record_manual-rag_delete_num_manual,out_manualswc_no_rag_end.begin()+j+1-rag_delete_num_manual);
                        rag_delete_num_manual += rag_num_manual;
                    }
                        rag_record_manual = j;
                }
                    rag_num_manual = 1;

            }

        }

        //删除auto毛刺
        for (V3DLONG j=0; j<nt_Auto.listNeuron.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = nt_Auto.listNeuron[j].x;
            S.y = nt_Auto.listNeuron[j].y;
            S.z = nt_Auto.listNeuron[j].z;
            S.n = nt_Auto.listNeuron[j].n;
            S.pn = nt_Auto.listNeuron[j].pn;
            S.r = nt_Auto.listNeuron[j].r;
            S.type = nt_Auto.listNeuron[j].type;

            out_autoswc_no_rag_end.push_back(S);

            rag_num_auto++;
            if(S.pn != (S.n-1) || j == nt_Auto.listNeuron.size()-1 )
            {
//                cout<<"auto S.n:"<<S.n<<endl;
                if(j != nt_Auto.listNeuron.size()-1)
                {
                    rag_num_auto--;
                    if(rag_num_auto <= thred_rag && rag_num_auto>0)
                    {
                        //删除小于阈值的毛刺
                        out_autoswc_no_rag_end.erase(out_autoswc_no_rag_end.begin()+rag_record_auto-rag_delete_num_auto,out_autoswc_no_rag_end.begin()+j-rag_delete_num_auto);
                        rag_delete_num_auto += rag_num_auto;
                    }
                    rag_record_auto = j;

                }else
                {
                    if(rag_num_auto <= thred_rag && rag_num_auto>0)
                    {
                        //删除小于阈值的毛刺
                        out_autoswc_no_rag_end.erase(out_autoswc_no_rag_end.begin()+rag_record_auto-rag_delete_num_auto,out_autoswc_no_rag_end.begin()+j+1-rag_delete_num_auto);
                        rag_delete_num_auto += rag_num_auto;
                    }
                    rag_record_auto = j;
                }

                rag_num_auto = 1;
            }

        }
        QList<NeuronSWC> out_autoswc_no_rag_sort,out_manualswc_no_rag_sort;
        SortSWC(out_autoswc_no_rag_end, out_autoswc_no_rag_sort ,VOID, 0);
        SortSWC(out_manualswc_no_rag_end, out_manualswc_no_rag_sort ,VOID, 0);

        //保存处理之后的SWC数据
        QStringList name2 = manual_swcfiles[i].split('/');
        QStringList name3 = auto_swcfiles[i].split('/');
        QString manualswc_name = name2[name2.count()-1];
        QString autoswc_name = name3[name3.count()-1];

        cout<<"start save..."<<endl;
        //output swc result
        QString manual_end_block = QString(outlist->at(0))+"/"+"method2_manual_norag_block"+"/"+manualswc_name;
        QString auto_end_block = QString(outlist->at(0))+"/"+"method2_auto_norag_block"+"/"+autoswc_name;

        export_list2file(out_manualswc_no_rag_sort,manual_end_block,manual_end_block);
        export_list2file(out_autoswc_no_rag_sort,auto_end_block,auto_end_block);
    }
}

bool delete_few_seg(const V3DPluginArgList &input,V3DPluginArgList &output)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

//    //output file
//    QString out_manual_swc = QString(outlist->at(0));
//    QString out_auto_swc = QString(outlist->at(1));
    vector<QString> few_seg_manual_txt,few_seg_auto_txt;

//    for(int i = 0;i < 70;i++)
    for(int i = 0;i < manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));
/***********************************************************************************
 去噪：删除结点个数小于thred不在block边缘的片段
 **********************************************************************************/
        cout<<"start delete few seg"<<endl;
        //read coordinate
        QStringList name1 = manual_swcfiles.at(i).split('/');
        QString neuron_name = name1[name1.count()-1];

        V3DLONG xl = neuron_name.split('_')[3].toInt();
        V3DLONG yl = neuron_name.split('_')[5].toInt();
        V3DLONG zl = neuron_name.split('_')[7].split('.')[0].toInt();
        V3DLONG xr = xl+64;
        V3DLONG yr = yl+64;
        V3DLONG zr = zl+64;

        QList<NeuronSWC> out_autoswc,out_manualswc;

        int thred = 3,thred_weight=0.1;
        vector<int> flag_seg_manual,flag_seg_auto;
        vector<int> num_seg_manual,num_seg_auto;
        int num = 0,record=0;
        //记录manual结点个数小于等于thred的片段
        for (V3DLONG j=0; j<nt_Manual.listNeuron.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = nt_Manual.listNeuron[j].x;
            S.y = nt_Manual.listNeuron[j].y;
            S.z = nt_Manual.listNeuron[j].z;
            S.n = nt_Manual.listNeuron[j].n;
            S.pn = nt_Manual.listNeuron[j].pn;
            S.r = nt_Manual.listNeuron[j].r;
            S.type = nt_Manual.listNeuron[j].type;

            out_manualswc.push_back(S);

            if(nt_Manual.listNeuron[j].pn<0 || j==(nt_Manual.listNeuron.size()-1))
            {
//                cout<<"flag_seg_manual:"<<j<<endl;
                flag_seg_manual.push_back(j);
            }

            if((record != flag_seg_manual.size() && flag_seg_manual.size()>1)  || j==(nt_Manual.listNeuron.size()-1))
            {
                record = flag_seg_manual.size();
                //最后两个都是-1
                if(j==(nt_Manual.listNeuron.size()-1) && nt_Manual.listNeuron[j].pn<0 )
                {
                    num_seg_manual.push_back(num);
                    num_seg_manual.push_back(1);
                }
                //最后一个是-1
                else if(j==(nt_Manual.listNeuron.size()-1))
                {
                    num_seg_manual.push_back(num+1);
                }else
                {
                    num_seg_manual.push_back(num);
                }
                num=0;
            }
            num++;
        }

        num = 0,record=0;
        //记录auto结点个数小于等于thred的片段
        for (V3DLONG j=0; j<nt_Auto.listNeuron.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = nt_Auto.listNeuron[j].x;
            S.y = nt_Auto.listNeuron[j].y;
            S.z = nt_Auto.listNeuron[j].z;
            S.n = nt_Auto.listNeuron[j].n;
            S.pn = nt_Auto.listNeuron[j].pn;
            S.r = nt_Auto.listNeuron[j].r;
            S.type = nt_Auto.listNeuron[j].type;

            out_autoswc.push_back(S);

            if(nt_Auto.listNeuron[j].pn<0 || j==(nt_Auto.listNeuron.size()-1))
            {
//                cout<<"flag_seg_auto:"<<j<<endl;
                flag_seg_auto.push_back(j);
            }

            if((record != flag_seg_auto.size() && flag_seg_auto.size()>1) || j==(nt_Auto.listNeuron.size()-1))
            {
                record = flag_seg_auto.size();
                //最后两个都是-1
                if(j==(nt_Auto.listNeuron.size()-1) && nt_Auto.listNeuron[j].pn<0 )
                {
                    num_seg_auto.push_back(num);
                    num_seg_auto.push_back(1);
                }
                //最后一个是-1
                else if(j==(nt_Auto.listNeuron.size()-1))
                {
                    num_seg_auto.push_back(num+1);
                }else
                {
                    num_seg_auto.push_back(num);
                }
                num=0;
            }
            num++;
        }


        V3DLONG delete_num = 0;
        //删除manual节点个数小于等于thred的片段
        for(unsigned long long m=0;m<num_seg_manual.size();m++)
        {
//            cout<<"num_seg_manual "<<m<<':'<<num_seg_manual[m]<<endl;
            if(num_seg_manual[m]<=thred)
            {
                if((m+1)>=flag_seg_manual.size())
                {
                    if(detective_six_bound(out_manualswc[flag_seg_manual[m]-delete_num].x,out_manualswc[flag_seg_manual[m]-delete_num].y,out_manualswc[flag_seg_manual[m]-delete_num].z,xl,yl,zl,xr,yr,zr,thred_weight))
                    {
//                        cout<<"delete"<<endl;
                        few_seg_manual_txt.push_back(neuron_name);
                        //片段不在边缘区，应该删除
                        out_manualswc.pop_back();
                    }
                }else
                {
                    for(int n = flag_seg_manual[m]-delete_num;n<(flag_seg_manual[m+1]-delete_num);n++)
                    {
                        if(detective_six_bound(out_manualswc[n].x,out_manualswc[n].y,out_manualswc[n].z,xl,yl,zl,xr,yr,zr,thred_weight))
                        {
                            few_seg_manual_txt.push_back(neuron_name);
                            //片段不在边缘区，应该删除
                            if(flag_seg_manual[m+1] == out_manualswc.size()-1)
                            {
                                out_manualswc.erase(out_manualswc.begin()+flag_seg_manual[m]-delete_num,out_manualswc.end());
                            }else
                            {
                                out_manualswc.erase(out_manualswc.begin()+flag_seg_manual[m]-delete_num,out_manualswc.begin()+flag_seg_manual[m+1]-1-delete_num);
                                delete_num = delete_num + num_seg_manual[m];
                             }
                            break;
                        }
                    }
                }

            }
        }

        delete_num = 0;
        //删除auto节点个数小于等于thred的片段
        for(unsigned long long m=0;m<num_seg_auto.size();m++)
        {
//            cout<<"num_seg_auto "<<m<<':'<<num_seg_auto[m]<<endl;
            if(num_seg_auto[m]<=thred)
            {
                if((m+1)>=flag_seg_auto.size())
                {
//                    cout<<"m"<<m+1<<endl;
                    if(detective_six_bound(out_autoswc[flag_seg_auto[m]-delete_num].x,out_autoswc[flag_seg_auto[m]-delete_num].y,out_autoswc[flag_seg_auto[m]-delete_num].z,xl,yl,zl,xr,yr,zr,thred_weight))
                    {
                        few_seg_auto_txt.push_back(neuron_name);
                        //片段不在边缘区，应该删除
                        out_autoswc.pop_back();
                    }

                }else
                {
                    for(int n = flag_seg_auto[m]-delete_num;n<(flag_seg_auto[m+1]-delete_num);n++)
                    {

                        if(detective_six_bound(out_autoswc[n].x,out_autoswc[n].y,out_autoswc[n].z,xl,yl,zl,xr,yr,zr,thred_weight))
                        {
                            few_seg_auto_txt.push_back(neuron_name);
                            if(flag_seg_auto[m+1] == out_autoswc.size()-1 )
                            {
                                out_autoswc.erase(out_autoswc.begin()+flag_seg_auto[m]-delete_num,out_autoswc.begin()+flag_seg_auto[m+1]+1-delete_num);
                            }else
                            {
                                out_autoswc.erase(out_autoswc.begin()+flag_seg_auto[m]-delete_num,out_autoswc.begin()+flag_seg_auto[m+1]-delete_num);
                                delete_num = delete_num + num_seg_auto[m];
                            }
                            break;
                        }
                    }
                }

            }
        }

        QList<NeuronSWC> out_autoswc_noseg_sort,out_manualswc_noseg_sort;
        V3DLONG rootid = VOID;
        V3DLONG thres = 0;
        SortSWC(out_autoswc, out_autoswc_noseg_sort ,rootid, thres);
        SortSWC(out_manualswc, out_manualswc_noseg_sort ,rootid, thres);

        //保存处理之后的SWC数据
        QStringList name2 = manual_swcfiles[i].split('/');
        QStringList name3 = auto_swcfiles[i].split('/');
        QString manualswc_name = name2[name2.count()-1];
        QString autoswc_name = name3[name3.count()-1];

        cout<<"start save..."<<endl;
        //output swc result
        QString manual_end_block = QString(outlist->at(0))+"/"+"method2_manual_norag_block"+"/"+manualswc_name;
        QString auto_end_block = QString(outlist->at(0))+"/"+"method2_auto_norag_block"+"/"+autoswc_name;

        export_list2file(out_manualswc_noseg_sort,manual_end_block,manual_end_block);
        export_list2file(out_autoswc_noseg_sort,auto_end_block,auto_end_block);
    }
}

//
//去掉毛刺(节点个数很少的分支)和不在block边缘位置节点个数很少的片段
bool delete_rag_branch_and_few_seg(const V3DPluginArgList &input,V3DPluginArgList &output)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

//    //output file
//    QString out_manual_swc = QString(outlist->at(0));
//    QString out_auto_swc = QString(outlist->at(1));
    vector<QString> few_seg_manual_txt,few_seg_auto_txt;

//    for(int i = 0;i < 70;i++)
    for(int i = 0;i < manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));

/***********************************************************************************
 去毛刺：删掉分叉结点个数小于thred的毛刺
 **********************************************************************************/
        cout<<"start delete rag"<<endl;
        //删除manual毛刺
        QList<NeuronSWC> out_autoswc_no_rag_end,out_manualswc_no_rag_end;
        V3DLONG rag_num_manual=0,rag_num_auto=0,rag_record_manual=0,rag_record_auto=0,
                rag_delete_num_manual=0,rag_delete_num_auto=0;
        int thred_rag = 3;
        for (V3DLONG j=0; j<nt_Manual.listNeuron.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = nt_Manual.listNeuron[j].x;
            S.y = nt_Manual.listNeuron[j].y;
            S.z = nt_Manual.listNeuron[j].z;
            S.n = nt_Manual.listNeuron[j].n;
            S.pn = nt_Manual.listNeuron[j].pn;
            S.r = nt_Manual.listNeuron[j].r;
            S.type = nt_Manual.listNeuron[j].type;

            out_manualswc_no_rag_end.push_back(S);

            rag_num_manual++;

            if(j!=0 && (S.pn != (S.n-1)|| j == nt_Manual.listNeuron.size()-1 ))
            {
//                cout<<"S.n:"<<S.n<<' '<<S.pn<<endl;
                if(j != nt_Manual.listNeuron.size()-1)
                {
                    rag_num_manual--;
                    if(rag_num_manual <= thred_rag)
                    {
//                        cout<<"delete1 rag"<<endl;
//                        cout<<j<<' '<<rag_record_manual<<' '<<rag_num_manual<<endl;
                        //删除小于阈值的毛刺
                        out_manualswc_no_rag_end.erase(out_manualswc_no_rag_end.begin()+rag_record_manual-rag_delete_num_manual,out_manualswc_no_rag_end.begin()+j-rag_delete_num_manual);
                        rag_delete_num_manual += rag_num_manual;
                    }
                    rag_record_manual = j;

                }else
                {
                    if(rag_num_manual <= thred_rag)
                    {
//                        cout<<"delete2 rag"<<endl;
//                        cout<<j<<' '<<rag_record_manual<<' '<<rag_num_manual<<' '<<out_manualswc_no_few_seg_sort.size()<<endl;
                        //删除小于阈值的毛刺
                        out_manualswc_no_rag_end.erase(out_manualswc_no_rag_end.begin()+rag_record_manual-rag_delete_num_manual,out_manualswc_no_rag_end.begin()+j+1-rag_delete_num_manual);
                        rag_delete_num_manual += rag_num_manual;
                    }
                    rag_record_manual = j;

                }
                rag_num_manual = 1;
            }

        }

        //删除auto毛刺
        for (V3DLONG j=0; j<nt_Auto.listNeuron.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = nt_Auto.listNeuron[j].x;
            S.y = nt_Auto.listNeuron[j].y;
            S.z = nt_Auto.listNeuron[j].z;
            S.n = nt_Auto.listNeuron[j].n;
            S.pn = nt_Auto.listNeuron[j].pn;
            S.r = nt_Auto.listNeuron[j].r;
            S.type = nt_Auto.listNeuron[j].type;

            out_autoswc_no_rag_end.push_back(S);

            rag_num_auto++;
            if(j!=0 && (S.pn != (S.n-1) || j == nt_Auto.listNeuron.size()-1 ))
            {
//                cout<<"auto S.n:"<<S.n<<endl;
                if(j != nt_Auto.listNeuron.size()-1)
                {

                    rag_num_auto--;
                    if(rag_num_auto <= thred_rag)
                    {
                        //删除小于阈值的毛刺
                        out_autoswc_no_rag_end.erase(out_autoswc_no_rag_end.begin()+rag_record_auto-rag_delete_num_auto,out_autoswc_no_rag_end.begin()+j-rag_delete_num_auto);
                        rag_delete_num_auto += rag_num_auto;
                    }
                    rag_record_auto = j;

                }else
                {
                    if(rag_num_auto <= thred_rag)
                    {
                        //删除小于阈值的毛刺
                        out_autoswc_no_rag_end.erase(out_autoswc_no_rag_end.begin()+rag_record_auto-rag_delete_num_auto,out_autoswc_no_rag_end.begin()+j+1-rag_delete_num_auto);
                        rag_delete_num_auto += rag_num_auto;
                    }
                    rag_record_auto = j;
                }
                rag_num_auto = 1;
            }

        }

        QList<NeuronSWC> out_autoswc_no_rag_sort,out_manualswc_no_rag_sort;
        SortSWC(out_autoswc_no_rag_end, out_autoswc_no_rag_sort ,VOID, 0);
        SortSWC(out_manualswc_no_rag_end, out_manualswc_no_rag_sort ,VOID, 0);

/***********************************************************************************
 去噪：删除结点个数小于thred不在block边缘的片段
 **********************************************************************************/
        cout<<"start delete few seg"<<endl;
        //read coordinate
        QStringList name1 = manual_swcfiles.at(i).split('/');
        QString neuron_name = name1[name1.count()-1];

        V3DLONG xl = neuron_name.split('_')[3].toInt();
        V3DLONG yl = neuron_name.split('_')[5].toInt();
        V3DLONG zl = neuron_name.split('_')[7].split('.')[0].toInt();
        V3DLONG xr = xl+64;
        V3DLONG yr = yl+64;
        V3DLONG zr = zl+64;

        QList<NeuronSWC> out_autoswc,out_manualswc;

        int thred = 3,thred_weight=0.1;
        vector<int> flag_seg_manual,flag_seg_auto;
        vector<int> num_seg_manual,num_seg_auto;
        int num = 0,record=0;
        //记录manual结点个数小于等于thred的片段
        for (V3DLONG j=0; j<out_manualswc_no_rag_sort.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = out_manualswc_no_rag_sort[j].x;
            S.y = out_manualswc_no_rag_sort[j].y;
            S.z = out_manualswc_no_rag_sort[j].z;
            S.n = out_manualswc_no_rag_sort[j].n;
            S.pn = out_manualswc_no_rag_sort[j].pn;
            S.r = out_manualswc_no_rag_sort[j].r;
            S.type = out_manualswc_no_rag_sort[j].type;

            out_manualswc.push_back(S);

            if(out_manualswc_no_rag_sort[j].pn<0 || j==(out_manualswc_no_rag_sort.size()-1))
            {
//                cout<<"flag_seg_manual:"<<j<<endl;
                flag_seg_manual.push_back(j);
            }

            if((record != flag_seg_manual.size() && flag_seg_manual.size()>1)  || j==(out_manualswc_no_rag_sort.size()-1))
            {
                record = flag_seg_manual.size();
                //最后两个都是-1
                if(j==(out_manualswc_no_rag_sort.size()-1) && out_manualswc_no_rag_sort[j].pn<0 )
                {
                    num_seg_manual.push_back(num);
                    num_seg_manual.push_back(1);
                }
                //最后一个是-1
                else if(j==(out_manualswc_no_rag_sort.size()-1))
                {
                    num_seg_manual.push_back(num+1);
                }else
                {
                    num_seg_manual.push_back(num);
                }
                num=0;
            }
            num++;
        }

        num = 0,record=0;
        //记录auto结点个数小于等于thred的片段
        for (V3DLONG j=0; j<out_autoswc_no_rag_sort.size();j++)
        {
            NeuronSWC S;
            //取swc文件的数据
            S.x = out_autoswc_no_rag_sort[j].x;
            S.y = out_autoswc_no_rag_sort[j].y;
            S.z = out_autoswc_no_rag_sort[j].z;
            S.n = out_autoswc_no_rag_sort[j].n;
            S.pn = out_autoswc_no_rag_sort[j].pn;
            S.r = out_autoswc_no_rag_sort[j].r;
            S.type = out_autoswc_no_rag_sort[j].type;

            out_autoswc.push_back(S);

            if(out_autoswc_no_rag_sort[j].pn<0 || j==(out_autoswc_no_rag_sort.size()-1))
            {
//                cout<<"flag_seg_auto:"<<j<<endl;
                flag_seg_auto.push_back(j);
            }

            if((record != flag_seg_auto.size() && flag_seg_auto.size()>1) || j==(out_autoswc_no_rag_sort.size()-1))
            {
                record = flag_seg_auto.size();
                //最后两个都是-1
                if(j==(out_autoswc_no_rag_sort.size()-1) && out_autoswc_no_rag_sort[j].pn<0 )
                {
                    num_seg_auto.push_back(num);
                    num_seg_auto.push_back(1);
                }
                //最后一个是-1
                else if(j==(out_autoswc_no_rag_sort.size()-1))
                {
                    num_seg_auto.push_back(num+1);
                }else
                {
                    num_seg_auto.push_back(num);
                }
                num=0;
            }
            num++;
        }


        V3DLONG delete_num = 0;
        //删除manual节点个数小于等于thred的片段
        for(unsigned long long m=0;m<num_seg_manual.size();m++)
        {
//            cout<<"num_seg_manual "<<m<<':'<<num_seg_manual[m]<<endl;
            if(num_seg_manual[m]<=thred)
            {
                if((m+1)>=flag_seg_manual.size())
                {
                    if(detective_six_bound(out_manualswc[flag_seg_manual[m]-delete_num].x,out_manualswc[flag_seg_manual[m]-delete_num].y,out_manualswc[flag_seg_manual[m]-delete_num].z,xl,yl,zl,xr,yr,zr,thred_weight))
                    {
//                        cout<<"delete"<<endl;
                        few_seg_manual_txt.push_back(neuron_name);
                        //片段不在边缘区，应该删除
                        out_manualswc.pop_back();
                    }
                }else
                {
                    for(int n = flag_seg_manual[m]-delete_num;n<(flag_seg_manual[m+1]-delete_num);n++)
                    {
                        if(detective_six_bound(out_manualswc[n].x,out_manualswc[n].y,out_manualswc[n].z,xl,yl,zl,xr,yr,zr,thred_weight))
                        {
                            few_seg_manual_txt.push_back(neuron_name);
                            //片段不在边缘区，应该删除
                            if(flag_seg_manual[m+1] == out_manualswc.size()-1)
                            {
                                out_manualswc.erase(out_manualswc.begin()+flag_seg_manual[m]-delete_num,out_manualswc.end());
                            }else
                            {
                                out_manualswc.erase(out_manualswc.begin()+flag_seg_manual[m]-delete_num,out_manualswc.begin()+flag_seg_manual[m+1]-1-delete_num);
                                delete_num = delete_num + num_seg_manual[m];
                             }
                            break;
                        }
                    }
                }

            }
        }

        delete_num = 0;
        //删除auto节点个数小于等于thred的片段
        for(unsigned long long m=0;m<num_seg_auto.size();m++)
        {
//            cout<<"num_seg_auto "<<m<<':'<<num_seg_auto[m]<<endl;
            if(num_seg_auto[m]<=thred)
            {
                if((m+1)>=flag_seg_auto.size())
                {
//                    cout<<"m"<<m+1<<endl;
                    if(detective_six_bound(out_autoswc[flag_seg_auto[m]-delete_num].x,out_autoswc[flag_seg_auto[m]-delete_num].y,out_autoswc[flag_seg_auto[m]-delete_num].z,xl,yl,zl,xr,yr,zr,thred_weight))
                    {
                        few_seg_auto_txt.push_back(neuron_name);
                        //片段不在边缘区，应该删除
                        out_autoswc.pop_back();
                    }

                }else
                {
                    for(int n = flag_seg_auto[m]-delete_num;n<(flag_seg_auto[m+1]-delete_num);n++)
                    {

                        if(detective_six_bound(out_autoswc[n].x,out_autoswc[n].y,out_autoswc[n].z,xl,yl,zl,xr,yr,zr,thred_weight))
                        {
                            few_seg_auto_txt.push_back(neuron_name);
                            if(flag_seg_auto[m+1] == out_autoswc.size()-1 )
                            {
                                out_autoswc.erase(out_autoswc.begin()+flag_seg_auto[m]-delete_num,out_autoswc.begin()+flag_seg_auto[m+1]+1-delete_num);
                            }else
                            {
                                out_autoswc.erase(out_autoswc.begin()+flag_seg_auto[m]-delete_num,out_autoswc.begin()+flag_seg_auto[m+1]-delete_num);
                                delete_num = delete_num + num_seg_auto[m];
                            }
                            break;
                        }
                    }
                }

            }
        }

        QList<NeuronSWC> out_autoswc_norag_noseg_sort,out_manualswc_norag_noseg_sort;
        V3DLONG rootid = VOID;
        V3DLONG thres = 0;
        SortSWC(out_autoswc, out_autoswc_norag_noseg_sort ,rootid, thres);
        SortSWC(out_manualswc, out_manualswc_norag_noseg_sort ,rootid, thres);
/***********************************************************************************
 保存文件
 **********************************************************************************/


        //保存处理之后的SWC数据
        QStringList name2 = manual_swcfiles[i].split('/');
        QStringList name3 = auto_swcfiles[i].split('/');
        QString manualswc_name = name2[name1.count()-1];
        QString autoswc_name = name3[name2.count()-1];

        cout<<"start save..."<<endl;
        //output swc result
        QString manual_end_block = QString(outlist->at(0))+"/"+"method2_manual_norag_nofew_block"+"/"+manualswc_name;
        QString auto_end_block = QString(outlist->at(0))+"/"+"method2_auto_norag_nofew_block"+"/"+autoswc_name;

        export_list2file(out_manualswc_norag_noseg_sort,manual_end_block,manual_end_block);
        export_list2file(out_autoswc_norag_noseg_sort,auto_end_block,auto_end_block);
    }

//    //记录经过去噪处理的神经元名字
//    QString manual_txt = QString(outlist->at(0))+"few_seg_manual.txt";
//    QString auto_txt = QString(outlist->at(0))+"few_seg_auto.txt";
//    save_nofrag_nofew_TXT(few_seg_manual_txt,manual_txt);
//    save_nofrag_nofew_TXT(few_seg_auto_txt,auto_txt);
}

//删除block中结点个数少于5个的树
bool delete_few_tree(const V3DPluginArgList &input,V3DPluginArgList &output)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //imput swc
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

    //output file
    QString out_txt = QString(outlist->at(0));

    //阈值
    int thred = 3;


    for(int i = 0;i < manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));

        QList<NeuronSWC> out_autoswc,out_manualswc;

        //遍历手工SWC
        V3DLONG n = 0;
        V3DLONG num = 0;
        for (V3DLONG j=0; j<nt_Manual.listNeuron.size();j++)
        {
            NeuronSWC S;

            if(nt_Manual.listNeuron[j].pn<0 || j==(nt_Manual.listNeuron.size()-1))
            {
                if(j==(nt_Manual.listNeuron.size()-1))
                {
                    num++;
                    if(num!=0 && num <= thred)
                    {
                        //删除小于thred个结点的树
                        for(V3DLONG k=n;k<=j;k++)
                        {
                            out_manualswc.pop_back();
                        }
                    }
                }else
                {
                    if(num!=0 && num <= thred)
                    {
                        //删除小于thred个结点的树
                        for(V3DLONG k=n;k<j;k++)
                        {
                            out_manualswc.pop_back();
                        }
                    }
                }
                n = j;
                num = 1;
            }
            else
            {
                num++;
            }

            //排除最后两行都是-1的情况
            if(!(nt_Manual.listNeuron[j].pn<0 && j==(nt_Manual.listNeuron.size()-1)))
            {

                //取swc文件的数据
                S.x = nt_Manual.listNeuron[j].x;
                S.y = nt_Manual.listNeuron[j].y;
                S.z = nt_Manual.listNeuron[j].z;
                S.n = nt_Manual.listNeuron[j].n;
                S.pn = nt_Manual.listNeuron[j].pn;
                S.r = nt_Manual.listNeuron[j].r;
                S.type = nt_Manual.listNeuron[j].type;

                out_manualswc.push_back(S);
            }

        }

        //遍历自动SWC
        n = 0;
        num = 0;

        for (V3DLONG j=0; j<nt_Auto.listNeuron.size();j++)
        {

            NeuronSWC S;

            if(nt_Auto.listNeuron[j].pn<0 || j==(nt_Auto.listNeuron.size()-1))
            {
                if(j==(nt_Auto.listNeuron.size()-1))
                {
                    num++;
                    if(num!=0 && num <= thred)
                    {
                        //删除小于thred个结点的树
                        for(V3DLONG k=n;k<=j;k++)
                        {
                            out_autoswc.pop_back();
                        }
                    }
                }else
                {
                    if(num!=0 && num <= thred)
                    {
                        //删除小于thred个结点的树
                        for(V3DLONG k=n;k<j;k++)
                        {
                            out_autoswc.pop_back();
                        }
                    }
                }
                n = j;
                num = 1;
            }
            else
            {
                num++;
            }

            //排除最后两行都是-1的情况
            if(!(nt_Auto.listNeuron[j].pn<0 && j==(nt_Auto.listNeuron.size()-1)))
            {
                //取swc文件的数据
                S.x = nt_Auto.listNeuron[j].x;
                S.y = nt_Auto.listNeuron[j].y;
                S.z = nt_Auto.listNeuron[j].z;
                S.n = nt_Auto.listNeuron[j].n;
                S.pn = nt_Auto.listNeuron[j].pn;
                S.r = nt_Auto.listNeuron[j].r;
                S.type = nt_Auto.listNeuron[j].type;

                out_autoswc.push_back(S);
            }

        }

        QList<NeuronSWC> out_autoswc_sort,out_manualswc_sort;
        V3DLONG rootid = VOID;
        V3DLONG thres = 0;
        SortSWC(out_autoswc, out_autoswc_sort ,rootid, thres);
        SortSWC(out_manualswc, out_manualswc_sort ,rootid, thres);

        //保存处理之后的SWC数据      
        QStringList name1 = manual_swcfiles[i].split('/');
        QStringList name2 = auto_swcfiles[i].split('/');
        QString manualswc_name = name1[name1.count()-1];
        QString autoswc_name = name2[name2.count()-1];

        cout<<"start save..."<<endl;
        //output swc result
        QString manual_end_block = QString(outlist->at(0))+"/"+"method1_manual_norag_block"+"/"+manualswc_name;
        QString auto_end_block = QString(outlist->at(0))+"/"+"method1_auto_norag_block"+"/"+autoswc_name;

        export_list2file(out_manualswc_sort,manual_end_block,manual_end_block);
        export_list2file(out_autoswc_sort,auto_end_block,auto_end_block);
    }

    cout<<"end..."<<endl;
}


void distance(V3DPluginCallback2 &callback,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //input swc dir
    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList auto_swclist = importFileList_addnumbersort(QString(inlist->at(1)));
    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

    //output distance txt
    QString out_distxt = QString(outlist->at(0));

    vector<V3DLONG> dis_list;
    for(int i=0;i<manual_swcfiles.size();i++)
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));

        V3DLONG len_Manual = nt_Manual.listNeuron.size();
        V3DLONG len_auto= nt_Auto.listNeuron.size();
        V3DLONG len = (nt_Manual.listNeuron.size()>nt_Auto.listNeuron.size())?nt_Auto.listNeuron.size():nt_Manual.listNeuron.size();
        V3DLONG dis = 0;

        for(int j =0; j < len; j++)
        {
            dis += MHDIS(nt_Manual.listNeuron[j],nt_Auto.listNeuron[j]);
        }
        dis_list.push_back(dis);
    }
    out_distxt = out_distxt + "/" + "distance.txt";
    vector<V3DLONG> ns;
    export_TXT(dis_list,out_distxt);


}


bool export_list2file(QList<NeuronSWC> & lN, QString fileSaveName, QString fileOpenName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);
    myfile<<"# generated by Vaa3D Plugin sort_neuron_swc"<<endl;
    myfile<<"# source file(s): "<<fileOpenName<<endl;
    myfile<<"# id,type,x,y,z,r,pid"<<endl;
    for (V3DLONG i=0;i<lN.size();i++)
        myfile << lN.at(i).n <<" " << lN.at(i).type << " "<< lN.at(i).x <<" "<<lN.at(i).y << " "<< lN.at(i).z << " "<< lN.at(i).r << " " <<lN.at(i).pn << "\n";

    file.close();
    cout<<"swc file "<<fileSaveName.toStdString()<<" has been generated, size: "<<lN.size()<<endl;
    return true;
};

QStringList importFileList_addnumbersort(const QString & curFilePath)
{
    QStringList myList;
    myList.clear();
    // get the iamge files namelist in the directory
    QStringList imgSuffix;
    imgSuffix<<"*.swc"<<"*.eswc"<<"*.SWC"<<"*.ESWC"<<"*.marjer";

    QDir dir(curFilePath);
    if(!dir.exists())
    {
        cout <<"Cannot find the directory";
        return myList;
    }
    foreach(QString file, dir.entryList()) // (imgSuffix, QDir::Files, QDir::Name))
    {
        myList += QFileInfo(dir, file).absoluteFilePath();
    }
    //print filenames
    foreach(QString qs, myList) qDebug() << qs;
    return myList;
}


bool export_TXT(vector<V3DLONG> &vec,QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<"distance#"<<endl;
    V3DLONG  p_pt=0;
    for (int i=0;i<vec.size(); i++)
    {
        //then save
        p_pt = vec.at(i);
        //myfile << p_pt->x<<"       "<<p_pt->y<<"       "<<p_pt->z<<"       "<<p_pt->signal<<endl;
        myfile <<i<<"\t"<< p_pt<<"\t"<<endl;
    }

    file.close();
    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec.size()<<endl;
    return true;
}

bool neuron_tree_num_TXT(vector<int> &vec1, vector<int> &vec2, vector<QString> &name, QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<"num"<<"\t"<<"neuron_name"<<"\t"<<"manual_tree_num"<<"\t"<<"auto_tree_num"<<endl;
//    V3DLONG  p_pt=0;
    for (int i=0;i<vec1.size(); i++)
    {
        //then save
//        p_pt = vec.at(i);
        myfile <<i<<"\t"<<name.at(i)<<"\t"<< vec1.at(i)<<"\t"<<vec2.at(i)<<endl;
    }

    file.close();
    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec1.size()<<endl;
    return true;
}


bool neuron_num_TXT(vector<V3DLONG> &vec,vector<QString> &name,QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<"num"<<"\t"<<"neuron_name"<<"neuron_num"<<endl;
//    V3DLONG  p_pt=0;
    for (int i=0;i<vec.size(); i++)
    {
        //then save
//        p_pt = vec.at(i);
        myfile <<i<<"\t"<<name.at(i)<<"\t"<< vec.at(i)<<"\t"<<endl;
    }

    file.close();
    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec.size()<<endl;
    return true;
}

//保存两个神经元重构的结果在一个block中的情况，也就是两个神经元重构的结果很接近
bool neuron_overlap_TXT(vector<V3DLONG> &vec1,vector<V3DLONG> &vec2,vector<V3DLONG> &vec3,vector<V3DLONG> &vec4,QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<QString::fromLocal8Bit("序号")<<"\t"<<QString::fromLocal8Bit("与金标准对应的神经元")<<"\t"<<QString::fromLocal8Bit("另外一个神经元")<<"\t"<<endl;
//    V3DLONG  p_pt=0;
    for (int i=0;i<vec1.size(); i++)
    {
        //then save
//        p_pt = vec.at(i);
        myfile <<i<<"\t"<< vec1.at(i)<<"\t"<<vec2.at(i)<<"\t"<<vec3.at(i)<<"\t"<<vec4.at(i)<<endl;
    }

    file.close();
//    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec.size()<<endl;
    return true;
}
//保存两个神经元重构的结果在一个block中的情况，也就是两个神经元重构的结果很接近
bool neuron_overlap_TXT2(vector<overlap_neuron> &vec1,QString fileSaveName)
{
    QFile file(fileSaveName);
    if (!file.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;
    QTextStream myfile(&file);

    myfile<<QString::fromLocal8Bit("序号")<<"\t"<<QString::fromLocal8Bit("与金标准对应的神经元")<<"\t"
         <<QString::fromLocal8Bit("另外一个神经元")<<"\t"<<QString::fromLocal8Bit("block序号")<<endl;
//    V3DLONG  p_pt=0;
    for (int i=0;i<vec1.size(); i++)
    {
        //then save
//        p_pt = vec.at(i);
        myfile <<i<<"\t"<<vec1[i].manual_file_name<<"\t"<<"\t"<<vec1[i].auto_file_name<<"\t"<<"\t"<<vec1[i].block_num<<endl;
    }

    file.close();
//    cout<<"txt file "<<fileSaveName.toStdString()<<" has been generated, size: "<<vec.size()<<endl;
    return true;
}

void get_blocks(V3DPluginCallback2 &callbacj,const V3DPluginArgList &input,V3DPluginArgList &output,QWidget *parent)
{
    vector<char*>* inlist = (vector<char*>*)(input.at(0).p);   //
    vector<char*>* outlist = (vector<char*>*)(output.at(0).p);
    vector<char*>* paralist = NULL;

    //input swc dir
    QString s= QString(inlist->at(0)) + "/" + "manual";
//    QStringList manual_swclist = importFileList_addnumbersort(QString(inlist->at(0)));
    QStringList manual_swclist = importFileList_addnumbersort(s);
    s = QString(inlist->at(0)) + "/" + "auto";
    QStringList auto_swclist = importFileList_addnumbersort(s);

    vector<QString> manual_swcfiles;
    vector<QString> auto_swcfiles;
    for(int i = 2;i < manual_swclist.size();i++)
        manual_swcfiles.push_back(manual_swclist.at(i));
    for(int i = 2;i < auto_swclist.size();i++)
        auto_swcfiles.push_back(auto_swclist.at(i));
    cout<<"manual_swcfiles : "<<manual_swcfiles.size()<<endl;
    cout<<"auto_swcfiles : "<<auto_swcfiles.size()<<endl;

    //output swc result
    s = QString(inlist->at(0)) + "/" + "manual_block";
    QString out_manualswc = QString(s);
    s = QString(inlist->at(0)) + "/" + "auto_block";
    QString out_autoswc = QString(s);

    //resample
    for(int i = 0;i < 1;i++)//manual_swcfiles.size()
    {
        //read swc
        cout<<"Processing "<<i+1<<"/"<<manual_swcfiles.size()<<" done!"<<endl;
        NeuronTree nt_Manual = readSWC_file(manual_swcfiles.at(i));
        NeuronTree nt_Auto = readSWC_file(auto_swcfiles.at(i));
        cout<<"nt_Manual.listNeuron = "<<nt_Manual.listNeuron.size()<<endl;
        cout<<"nt_Auto.listNeuron = "<<nt_Auto.listNeuron.size()<<endl;

        //save name
        QString manual_firstname = manual_swcfiles.at(i).split(".").first().split("/").at(5);
//        cout<<"manual_firstname:"<<manual_firstname.toStdString()<<endl;
        QString auto_firstname = auto_swcfiles.at(i).split(".").first().split("/").at(5);


        //resample
        NeuronTree nt_resample_manual,nt_resample_auto;
        double resample_step = 1;
        nt_resample_manual=resample(nt_Manual,resample_step);
        nt_resample_auto=resample(nt_Auto,resample_step );
        cout<<"nt_resample_manual.size = "<<nt_resample_manual.listNeuron.size()<<endl;
        cout<<"nt_resample_auto.size = "<<nt_resample_auto.listNeuron.size()<<endl;

        int step1 = 100;
        double l_x = 32;
        double l_y = 32;
        double l_z = 16;

        V3DLONG count = 0;
        V3DLONG tem = (nt_resample_manual.listNeuron.size()>nt_resample_auto.listNeuron.size())?nt_resample_auto.listNeuron.size():nt_resample_manual.listNeuron.size();
        for(V3DLONG j =0;j < tem;j=j+step1)
        {
            //count
            count++;

            QList<ImageMarker> marker_manual,marker_auto;
            marker_manual.clear();
            marker_auto.clear();
            ImageMarker m;
            m.x = nt_resample_manual.listNeuron[j].x;
            m.y = nt_resample_manual.listNeuron[j].y;
            m.z = nt_resample_manual.listNeuron[j].z;
            marker_manual.push_back(m);

            s = QString(inlist->at(0)) + "/" + "marker";
            QString marker_file_manual = s;
            writeMarker_file(marker_file_manual + "/"  + "m_"+manual_firstname +"_"+QString::number(count-1)+".marker",marker_manual);


            LocationSimple t_manual, t_auto;
            t_manual.x = nt_resample_manual.listNeuron[j].x;
            t_manual.y = nt_resample_manual.listNeuron[j].y;
            t_manual.z = nt_resample_manual.listNeuron[j].z;
            t_auto.x = nt_resample_auto.listNeuron[j].x;
            t_auto.y = nt_resample_auto.listNeuron[j].y;
            t_auto.z = nt_resample_auto.listNeuron[j].z;

            V3DLONG xb_manual = t_manual.x-l_x;
            V3DLONG xe_manual = t_manual.x+l_x-1;
            V3DLONG yb_manual = t_manual.y-l_y;
            V3DLONG ye_manual = t_manual.y+l_y-1;
            V3DLONG zb_manual = t_manual.z-l_z;
            V3DLONG ze_manual = t_manual.z+l_z-1;
            V3DLONG xb_auto = t_auto.x-l_x;
            V3DLONG xe_auto = t_auto.x+l_x-1;
            V3DLONG yb_auto = t_auto.y-l_y;
            V3DLONG ye_auto= t_auto.y+l_y-1;
            V3DLONG zb_auto = t_auto.z-l_z;
            V3DLONG ze_auto = t_auto.z+l_z-1;

            QList<NeuronSWC> outswc_manual,outswc_auto;
//神经元个数固定
            int nt_num = 20;
            for(V3DLONG l= j;l < (j+nt_num);l++)
            {
                if((j+nt_num)>nt_resample_manual.listNeuron.size())
                    break;
                NeuronSWC S;
                    S.x = nt_resample_manual.listNeuron[l].x;
                    S.y = nt_resample_manual.listNeuron[l].y;
                    S.z = nt_resample_manual.listNeuron[l].z;
                    S.n = nt_resample_manual.listNeuron[l].n;
                    S.pn = nt_resample_manual.listNeuron[l].pn;
                    S.r = nt_resample_manual.listNeuron[l].r;
                    S.type = nt_resample_manual.listNeuron[l].type;

                    outswc_manual.push_back(S);
             }

            for(V3DLONG l= j;l < (j+nt_num);l++)
            {
                if((j+nt_num)>nt_resample_auto.listNeuron.size())
                    break;
                NeuronSWC S;
                    S.x = nt_resample_auto.listNeuron[l].x;
                    S.y = nt_resample_auto.listNeuron[l].y;
                    S.z = nt_resample_auto.listNeuron[l].z;
                    S.n = nt_resample_auto.listNeuron[l].n;
                    S.pn = nt_resample_auto.listNeuron[l].pn;
                    S.r = nt_resample_auto.listNeuron[l].r;
                    S.type = nt_resample_auto.listNeuron[l].type;

                    outswc_auto.push_back(S);
             }

////长宽高固定
//            for(V3DLONG l= 0;l < nt_resample_auto.listNeuron.size();l++)
//            {
//                NeuronSWC S;
//                if(nt_resample_manual.listNeuron[l].x<xe_manual && nt_resample_manual.listNeuron[l].x>xb_manual && nt_resample_manual.listNeuron[l].y<ye_manual && nt_resample_manual.listNeuron[l].y>yb_manual && nt_resample_manual.listNeuron[l].z<ze_manual && nt_resample_manual.listNeuron[l].z>zb_manual)
//                {
//                    S.x = nt_resample_manual.listNeuron[l].x;
//                    S.y = nt_resample_manual.listNeuron[l].y;
//                    S.z = nt_resample_manual.listNeuron[l].z;
//                    S.n = nt_resample_manual.listNeuron[l].n;
//                    S.pn = nt_resample_manual.listNeuron[l].pn;
//                    S.r = nt_resample_manual.listNeuron[l].r;
//                    S.type = nt_resample_manual.listNeuron[l].type;

//                    outswc_manual.push_back(S);
//                }
//            }

//            for(V3DLONG l= 0;l < nt_resample_auto.listNeuron.size();l++)
//            {
//                NeuronSWC S;
//                if(nt_resample_auto.listNeuron[l].x<xe_auto && nt_resample_auto.listNeuron[l].x>xb_auto && nt_resample_auto.listNeuron[l].y<ye_auto && nt_resample_auto.listNeuron[l].y>yb_auto && nt_resample_auto.listNeuron[l].z<ze_auto && nt_resample_auto.listNeuron[l].z>zb_auto)
//                {
//                    S.x = nt_resample_auto.listNeuron[l].x;
//                    S.y = nt_resample_auto.listNeuron[l].y;
//                    S.z = nt_resample_auto.listNeuron[l].z;
//                    S.n = nt_resample_auto.listNeuron[l].n;
//                    S.pn = nt_resample_auto.listNeuron[l].pn;
//                    S.r = nt_resample_auto.listNeuron[l].r;
//                    S.type = nt_resample_auto.listNeuron[l].type;

//                    outswc_auto.push_back(S);
//                }
//            }

            QString outswc_file_manual,outswc_file_auto;
            outswc_file_manual = out_manualswc+"/" + "b_"+manual_firstname +"_"+QString::number(count-1)+".swc";
//            cout<<"outswc_file_manual:"<<outswc_file_manual.toStdString()<<endl;
            outswc_file_auto = out_autoswc+"/"+"b_"+auto_firstname +"_"+QString::number(count-1)+".swc";
//            cout<<"outswc_file_auto:"<<outswc_file_auto.toStdString()<<endl;

            export_list2file(outswc_manual,outswc_file_manual,outswc_file_manual);
            export_list2file(outswc_auto,outswc_file_auto,outswc_file_auto);

        }

    }



}




