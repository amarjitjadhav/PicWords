# PicWords

PicWords: Render a Picture by Packing Keywords

  PicWords can be considered as a kind of non-photorealistic rendering. Given a source picture, we first generate 
its silhouette, which is a binary image containing a Yang part and a Yin part. Yang part is for keywords placing while 
the Yin part can be ignored. Next, the Yang part is further over segmented into small patches, each of which serves as 
a container for one keyword. To make sure that more important keywords are put into more salient and larger image patches, 
we rank both the patches and keywords and construct a correspondence between the patch list and keyword list. Then, mean 
value coordinates method is used for the keyword-patch warping.

![alt text] (https://github.com/amarjitjadhav/PicWords/blob/master/implementation/backup/1.png)

In this section, we give an overview of the whole PicWords system. The whole system contains three modules: 
1) Picture Module  
2) keywords Module 
3) Picture & Keywords Module. - I am working on this currently

Reference - PicWords: Render a Picture by Packing Keywords.Zhenzhen Hu, Si Liu, Jianguo Jiang, Richang Hong, Meng Wang, and 
  Shuicheng Yan, 
