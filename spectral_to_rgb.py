import tensorflow as tf
import constants

from PIL import Image
from optics.sensor_srfs import SRF_OUTPUT_SIZE_LAMBDA
from log import Logger
from optics.camera import Camera
from optics.sensor import Sensor

from optics.diffractive_optical_element import Rank1HeightMapDOELayer
from optics.diffractive_optical_element import HeightMapDOELayer
from optics.diffractive_optical_element import QuantizedHeightMapDOELayer
from optics.diffractive_optical_element import QuadSymmetricQuantizedHeightMapDoeLayer
from optics.diffractive_optical_element import RotationallySymmetricQuantizedHeightMapDOELayer
from optics.util import laplace_l1_regularizer

from constants import MATERIAL_REFRACTIVE_INDEX_FUNCS

from util.data.data_utils import *


from util.data.dataset_loader import *

class  sepctral_to_rgb(tf.keras.Model):
    def __init__(self,image_patch_size,sensor_distance,wavelength_to_refractive_index_func_name,wave_resolution,
                 sample_interval,input_channel_num,doe_layer_type,depth_bin,
                 wave_length_list = constants.wave_length_list_400_700nm,default_optimizer_learning_rate_args = None,
                 reconstruction_network_type = None,reconstruction_network_args = None,
                 network_optimizer_learning_rate_args = None, use_psf_fixed_camera = False,
                 srf_type = None,doe_extra_args = None,height_map_noise = None,skip_optical_encoding = False,
                 use_extra_optimizer = False,extra_optimizer_learning_rate_args = None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        Logger.i("Wavelength list used ",wave_length_list)
        
        if doe_extra_args is None :
            doe_extra_args = {}
        self.image_patch_size = image_patch_size
        self.sensor_distance = sensor_distance
        
        self.wave_length_list = wave_length_list
        self.sample_interval = sample_interval
        self.wave_resolution = wave_resolution
        self.input_channel_num = input_channel_num
        
        self.doe_layer_type = doe_layer_type
        self.doe_layer = None
        
        self.srf_type = srf_type
        self._input_shape = None
        self.height_map_noise = height_map_noise
        
        
        self.wavelength_to_refractive_index_func = \
            MATERIAL_REFRACTIVE_INDEX_FUNCS[wavelength_to_refractive_index_func_name]
        assert self.wavelength_to_refractive_index_func is not None, \
            "Unsupported `doe_material` argument. It should be in: " +str(MATERIAL_REFRACTIVE_INDEX_FUNCS.key())
            
            
            
        doe_general_args  ={
            "wave_length_list": wave_length_list,
            "wavelength_to_refractive_index_func":self.wavelength_to_refractive_index_func,
            "height_map_initializer":None,
            "height_tolerance":height_map_noise,
            
            }
        Logger.i("\n\n==============>DOE Args<===============\n  > General:\n", doe_general_args,
                         "\n  > Extra:\n", doe_extra_args, "==============<DOE Args>===============\n\n")
        
        
        if doe_layer_type == "rank1":
            self.doe_layer = Rank1HeightMapDOELayer(**doe_general_args,height_map_regularizer = None,**doe_extra_args)
        elif doe_layer_type == 'htmp':
            self.doe_layer = HeightMapDOELayer(**doe_general_args,                              		  height_map_regularizer=laplace_l1_regularizer(scale=0.1),**doe_extra_args)
        elif doe_layer_type == 'htmp-quant':
            self.doe_layer = QuantizedHeightMapDOELayer(**doe_general_args,**doe_extra_args)
        elif doe_layer_type == 'htmp-quant-quad':
            self.doe_layer = QuadSymmetricQuantizedHeightMapDoeLayer(**doe_general_args,**doe_extra_args)
        elif doe_layer_type == 'htmp-quant-sym':
            self.doe_layer = RotationallySymmetricQuantizedHeightMapDOELayer(**doe_general_args,**doe_extra_args)
            
        assert self.doe_layer is not None, "Problems occurred and the DOE layer is None. Check your settings."
        
        sensor = None
        
        if srf_type is not None:
        	sensor = Sensor(srf_type=srf_type)
        
        self.optical_system = Camera(wave_resolution=self.wave_resolution,
                                    wave_length_list=self.wave_length_list,
                                    sensor_distance=self.sensor_distance,
                                    sensor_resolution=(self.image_patch_size, self.image_patch_size),
                                    sensor=sensor,
                                    input_sample_interval=self.sample_interval,
                                    doe_layer=self.doe_layer,
                                    input_channel_num=self.input_channel_num,
                                    depth_list=depth_bin, should_use_planar_incidence=False,
                                    should_depth_dependent=False).done()
        self.model_description = "DOE{}_SpItv{}_SsDst{}_ScDst{}_WvRes{}_ImgSz{}_SRF{}" \
            .format(doe_layer_type, sample_interval, sensor_distance, depth_bin[0], wave_resolution[0],
                    image_patch_size, srf_type)
        
        def call(selfs,input,**kwargs):
            x = self.optical_system(inputs, training=training, testing=testing)
            encoded_image = tf.image.encode_jpeg(x)
            tf.summary.image(name="SensorImage", data=x, max_outputs=1)
            print("test it is seccessful")
            return encoded_image
            
upsample_rate = 2
image_patch_size = 512
doe_resolution = 512 * upsample_rate
doe_layer_type = "htmp-quant-sym"
srf_type = "rgb"
network_input_size = SRF_OUTPUT_SIZE_LAMBDA[srf_type](image_patch_size)
batch_size = 4
step_per_epoch = 1672 // batch_size       

controlled_model_args = {
    "image_patch_size": 1672//4, "sensor_distance": 50e-3,
    "wavelength_to_refractive_index_func_name": None,
    "sample_interval": 8e-6 / 2,
    "wave_resolution": (512*2 , 512*2),
    "input_channel_num": 31, "depth_bin": [1],
    "doe_layer_type": "htmp-quant-sym",
    "srf_type": srf_type,
    "default_optimizer_learning_rate_args": {
        "initial_learning_rate": 0.01, "decay_steps": 500, "decay_rate": 0.8, "name": "default_opt_lr"},
    "network_optimizer_learning_rate_args": {
        "initial_learning_rate": 0.001, "decay_steps": 500, "decay_rate": 0.8, "name": "network_opt_lr"},
    "reconstruction_network_type": "res_block_u_net",
    "reconstruction_network_args": {
        "filter_root": 32, "depth": 7, "output_channel": 31, "input_size": network_input_size,
        "activation": 'elu', "batch_norm": True, "batch_norm_after_activation": False,
        "final_activation": 'sigmoid', "net_num": 1, "extra_upsampling": (srf_type == "rggb4"),
        "remove_first_long_connection": False, "channel_attention": False,
        "kernel_initializer": 'he_uniform', "final_kernel_initializer": 'glorot_uniform'
    },
    "height_map_noise": None
}


def Parameter_init(doe_material="BK7", with_doe_noise=True, quantization_level=256, quantize_at_test_only=False,
          alpha_blending=False, adaptive_quantization=False, checkpoint=None, continue_training=False,
          tag=None, sensor_distance_mm=30, scene_depth_m=5):
        controlled_model_args["wavelength_to_refractive_index_func_name"] = doe_material
        controlled_model_args["height_map_noise"] = 20e-9
        controlled_model_args["sensor_distance"] = sensor_distance_mm * 1e-3
        controlled_model_args["depth_bin"] = [scene_depth_m]
        controlled_model_args["doe_extra_args"] = {
        "quantization_level_cnt": quantization_level,
        "quantize_at_test_only": quantize_at_test_only,
        "adaptive_quantization": adaptive_quantization,
        "alpha_blending": alpha_blending,
        "step_per_epoch": step_per_epoch,
        "alpha_blending_start_epoch": 5,
        "alpha_blending_end_epoch": 40
    }

def flat_map_1024_overlapped_patches_from_ICVL_dataset(_img):
    flat_map_operation_list = [tf.image.resize(images=_img, size=[1024, 1024], method=ResizeMethod.BILINEAR)]
    #third_height = 464
    #third_width = 434
    #for i in range(0, 1392, third_height):
        #for j in range(0, 1300, third_width):
            #flat_map_operation_list.append(safe_crop_to_bounding_box(_img, i, j, 512, 512))
    return tf.data.Dataset.from_tensor_slices(flat_map_operation_list)

def load_icvl_full_mat_1024(dataset_dir, cache_name = "first_image", verbose=True):
    return load_icvl_full_mat(dataset_dir, verbose=verbose).flat_map(
        flat_map_1024_overlapped_patches_from_ICVL_dataset).cache(cache_name)
        

if __name__ == "__main__":
    dataset_dir = r'.\input_image'
    mat_dataset = load_icvl_full_mat_1024(dataset_dir)
    Parameter_init()
    input_image = mat_dataset
    model=sepctral_to_rgb(**controlled_model_args)
    output_image = model(input = mat_dataset)
    #encoded_image = tf.image.encode_jpeg(output_image)
    #encoded_image.show()
    #encoded_image.save("./encoded_image.jpg",'JPEG')
    #encoded_image.save("./encoded_image.jpg",'JPEG')
    print("aaa")























































    
