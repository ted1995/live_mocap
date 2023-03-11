import bpy
import math

input_fbx_path="./model/Rin.fbx"
output_fbx_path="./t1.fbx"

bpy.ops.import_scene.fbx(filepath=input_fbx_path)#导入fbx文件
bpy.context.scene.tool_settings.lock_object_mode = False #关闭 Edit->Lock Objects modes 选项
def createKeyFrame():
    ob = bpy.data.objects['Armature'] 
    bpy.context.view_layer.objects.active = ob #相当于鼠标左键选中
    bpy.ops.object.mode_set(mode='POSE') #切换为pose更改模式
    hip=ob.pose.bones['Hips'] #选中其中一块骨骼，根据自己模型中骨骼的名称，名称可以在Outliner(大纲)视图找到
	#对骨骼进行旋转
    hip.rotation_mode = 'XYZ'
    # select axis in ['X','Y','Z']  <--bone local
    axis = 'Z'
    angle = 120
    hip.rotation_euler.rotate_axis(axis, math.radians(angle))
    bpy.ops.object.mode_set(mode='OBJECT')
    #insert a keyframe
    hip.keyframe_insert(data_path="rotation_euler" ,frame=1) 
createKeyFrame()   
bpy.ops.export_scene.fbx(filepath=output_fbx_path) #导出操作之后的模型
