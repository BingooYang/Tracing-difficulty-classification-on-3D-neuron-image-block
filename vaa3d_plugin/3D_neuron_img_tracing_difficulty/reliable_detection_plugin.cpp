/* reliable_detection_plugin.cpp
 * This is a test plugin, you can use it as a demo.
 * 2020-6-12 : by YourName
 */
 
#include "v3d_message.h"
#include <vector>
#include "basic_surf_objs.h"

#include "reliable_detection_plugin.h"
Q_EXPORT_PLUGIN2(reliable_detection, TestPlugin);

using namespace std;


void reconstruction_func(V3DPluginCallback2 &callback, QWidget *parent, input_PARA &PARA, bool bmenu);
 
QStringList TestPlugin::menulist() const
{
	return QStringList() 
		<<tr("tracing_menu")
		<<tr("about");
}

QStringList TestPlugin::funclist() const
{
	return QStringList()
        <<tr("get_blocks")
        <<tr("neuron_distance")
        <<tr("block_tree")
        <<tr("delete_few_tree")
        <<tr("block_tree_num")
        <<tr("delete_rag_few_seg")
        <<tr("blocks_distance")
		<<tr("help");
}

void TestPlugin::domenu(const QString &menu_name, V3DPluginCallback2 &callback, QWidget *parent)
{
	if (menu_name == tr("tracing_menu"))
	{
        bool bmenu = true;
        input_PARA PARA;
        reconstruction_func(callback,parent,PARA,bmenu);

	}
	else
	{
		v3d_msg(tr("This is a test plugin, you can use it as a demo.. "
			"Developed by YourName, 2020-6-12"));
	}
}

bool TestPlugin::dofunc(const QString & func_name, const V3DPluginArgList & input, V3DPluginArgList & output, V3DPluginCallback2 & callback,  QWidget * parent)
{
    cout<<"start.."<<endl;


    if (func_name == tr("get_blocks"))
	{
        vector<char*> * pinfiles = (input.size() >= 1) ? (vector<char*> *) input[0].p : 0;
        vector<char*> * pparas = (input.size() >= 2) ? (vector<char*> *) input[1].p : 0;
        vector<char*> infiles = (pinfiles != 0) ? * pinfiles : vector<char*>();
        vector<char*> paras = (pparas != 0) ? * pparas : vector<char*>();
        vector<char*>* outlist = (vector<char*>*)(output.at(0).p);

        if(infiles.empty())
        {
            fprintf (stderr, "Need input image. \n");
            return false;
        }

        //dir struct
        input_PARA P;
        //input dir
        P.autoswc_file = infiles[0];
        P.manualswc_file = infiles[1];
//        P.inimg_file = infiles[2];
        cout<<"autoSWC file : "<<P.autoswc_file.toStdString()<<endl;
        cout<<"manualSWC file : "<<P.manualswc_file.toStdString()<<endl;
//        cout<<"inimg file : "<<P.inimg_file.toStdString()<<endl;

        //set output dir
        QString outpath_name = QString(outlist->at(0));
        P.outimg_bad_block = outpath_name + "/" +"outimg_bad_block";
        P.outimg_good_block = outpath_name + "/" +"outimg_good_block";
        P.outautoswc_bad_block = outpath_name + "/" +"method2_auto_few_block";
        P.outautoswc_good_block = outpath_name + "/" +"method2_auto_many_block";
        P.outmanualswc_bad_block = outpath_name + "/" +"method2_manual_few_block";
        P.outmanualswc_good_block = outpath_name + "/" +"method2_manual_many_block";

//        P.outautoswc_bad_block = outpath_name + "/" +"OneToOne_outautoswc_few_block";
//        P.outautoswc_good_block = outpath_name + "/" +"OneToOne_outautoswc_many_block";
//        P.outmanualswc_bad_block = outpath_name + "/" +"OneToOne_outmanualswc_few_block";
//        P.outmanualswc_good_block = outpath_name + "/" +"OneToOne_outmanualswc_many_block";

        P.neuron_overlap_name = outpath_name;

        cout<<"start..."<<endl;
//        different_confidence_area_samples_method1(callback,P);
        different_confidence_area_samples_OneToOne(callback,P);

//        get_blocks(callback,input,output,parent);

    }else if(func_name == tr("neuron_distance"))
    {
        cout<<"start..."<<endl;
        //计算20个对应神经元之间的距离
        distance(callback,input,output,parent);
    }else if(func_name == tr("block_tree_type"))
    {
        cout<<"start block_tree..."<<endl;
        detective_neuron_num_type(callback,input,output,parent);
    }
    else if(func_name == tr("delete_few_tree"))
    {
        cout<<"start delete_few_tree..."<<endl;
        delete_few_tree(input,output);
    }else if (func_name == tr("block_tree_num"))
    {
        cout<<"start detective_neuron_tree_num..."<<endl;
        detective_neuron_tree_num(callback,input,output,parent);
    }
    else if (func_name == tr("delete_rag_few_seg"))
    {
        cout<<"start test..."<<endl;
        delete_rag_branch_and_few_seg(input,output);
//        delete_rag_branch(input,output);
    }else if(func_name == tr("blocks_distance"))
    {
        compute_blocks_distance(input,output);
    }
    else if (func_name == tr("help"))
    {

        ////HERE IS WHERE THE DEVELOPERS SHOULD UPDATE THE USAGE OF THE PLUGIN


		printf("**** Usage of reliable_detection tracing **** \n");
		printf("vaa3d -x reliable_detection -f tracing_func -i <inimg_file> -p <channel> <other parameters>\n");
        printf("inimg_file       The input image\n");
        printf("channel          Data channel for tracing. Start from 1 (default 1).\n");
        printf("outswc_file      Will be named automatically based on the input image file name, so you don't have to specify it.\n\n");
        printf("------------------------------------------------------------\n");
        printf("------------------------------------------------------------\n");
        printf("different_confidence_area_samples usage:\n");
        printf("windows:\n");
        printf("vaa3d_msvc.exe /x reliable_detection /f get_blocks /i <autoswc_file> <manualswc_file> <inimg_file> /o <output_file>\n");
        printf("<output_file> should include six flods：outimg_bad_block outimg_good_block outautoswc_bad_block outautoswc_good_block outmanualswc_bad_block outmanualswc_good_block\n\n");
        printf("detective_neuron_num usage:\n");
        printf("windows:\n");
        printf("vaa3d_msvc.exe /x reliable_detection /f block_tree /i <manualswc_file> <autoswc_file>  /o <output_file>\n");
        printf("delete_few_tree usage:\n");
        printf("windows:\n");
        printf("vaa3d_msvc.exe /x reliable_detection /f delete_few_tree /i <manualswc_file> <autoswc_file> /o <output_file>\n");

	}
	else return false;

	return true;
}

void reconstruction_func(V3DPluginCallback2 &callback, QWidget *parent, input_PARA &PARA, bool bmenu)
{
    unsigned char* data1d = 0;
    V3DLONG N,M,P,sc,c;
    V3DLONG in_sz[4];
    if(bmenu)
    {
        v3dhandle curwin = callback.currentImageWindow();
        if (!curwin)
        {
            QMessageBox::information(0, "", "You don't have any image open in the main window.");
            return;
        }

        Image4DSimple* p4DImage = callback.getImage(curwin);

        if (!p4DImage)
        {
            QMessageBox::information(0, "", "The image pointer is invalid. Ensure your data is valid and try again!");
            return;
        }


        data1d = p4DImage->getRawData();
        N = p4DImage->getXDim();
        M = p4DImage->getYDim();
        P = p4DImage->getZDim();
        sc = p4DImage->getCDim();

        bool ok1;

        if(sc==1)
        {
            c=1;
            ok1=true;
        }
        else
        {
            c = QInputDialog::getInteger(parent, "Channel",
                                             "Enter channel NO:",
                                             1, 1, sc, 1, &ok1);
        }

        if(!ok1)
            return;

        in_sz[0] = N;
        in_sz[1] = M;
        in_sz[2] = P;
        in_sz[3] = sc;


        PARA.inimg_file = p4DImage->getFileName();
    }
    else
    {
        int datatype = 0;
        if (!simple_loadimage_wrapper(callback,PARA.inimg_file.toStdString().c_str(), data1d, in_sz, datatype))
        {
            fprintf (stderr, "Error happens in reading the subject file [%s]. Exit. \n",PARA.inimg_file.toStdString().c_str());
            return;
        }
        if(PARA.channel < 1 || PARA.channel > in_sz[3])
        {
            fprintf (stderr, "Invalid channel number. \n");
            return;
        }
        N = in_sz[0];
        M = in_sz[1];
        P = in_sz[2];
        sc = in_sz[3];
        c = PARA.channel;
    }

    //main neuron reconstruction code

    //// THIS IS WHERE THE DEVELOPERS SHOULD ADD THEIR OWN NEURON TRACING CODE

    //Output
    NeuronTree nt;
	QString swc_name = PARA.inimg_file + "_reliable_detection.swc";
	nt.name = "reliable_detection";
    writeSWC_file(swc_name.toStdString().c_str(),nt);

    if(!bmenu)
    {
        if(data1d) {delete []data1d; data1d = 0;}
    }

    v3d_msg(QString("Now you can drag and drop the generated swc fle [%1] into Vaa3D.").arg(swc_name.toStdString().c_str()),bmenu);

    return;
}
