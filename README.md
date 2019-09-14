# ssd_circle-anchor

have change the lable box to a lable circle using diagonal as diameter. (in util.py, create data list)
modify the layer and filter shape to adjust new anchor. original anchor is labled as (x,y,w,h). new is (x,y,r)
disable some of original code since the circle coordinate is already centralied.( xy_to_cxcy，cxcy_to_xy)
using new function to determine the way of calculating overlap index


data augmentation part is sill left to be change.
the detect.py and eval.py  are from open-souce code and have not been modified yet (interface)


error:
right now the problem is when load the data the last line of dataset.py  (function : collate.py) image=torch.stack(images)
             TypeError: expected Tensor as element 0 in argument 0, but got Image
             
             i think it's the change of lable box coordinations (4 dim->3 dim) causes the mismatch of dimension. but i'm still trying to pinpoint it.
            
