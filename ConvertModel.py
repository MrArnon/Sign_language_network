import coremltools
output_labels=['0','1','2','3','4','5','6','7','8','9']
coreml_model = coremltools.converters.keras.convert('model_check.h5',input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=output_labels)
#coremltools.utils.save_spec(coreml_model, 'my_model.mlmodel')
coreml_model.save('sign_lang.mlmodel')
