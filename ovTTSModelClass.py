import openvino as ov
import os
class OVBarkTextEncoder:
    def __init__(self, core, device, model_path1, model_path2):
        self.compiled_model1 = core.compile_model(model_path1, device)
        self.compiled_model2 = core.compile_model(model_path2, device)

    def __call__(self, input_ids, past_kv=None):
        if past_kv is None:
            outputs = self.compiled_model1(input_ids, share_outputs=True)
        else:
            outputs = self.compiled_model2([input_ids, *past_kv], share_outputs=True)
        logits, kv_cache = self.postprocess_outputs(outputs, past_kv is None)
        return logits, kv_cache

    def postprocess_outputs(self, outs, is_first_stage):
        net_outs = self.compiled_model1.outputs if is_first_stage else self.compiled_model2.outputs
        logits = outs[net_outs[0]]
        kv_cache = []
        for out_tensor in net_outs[1:]:
            kv_cache.append(outs[out_tensor])
        return logits, kv_cache


class OVBarkEncoder:
    def __init__(self, core, device, model_path):
        self.compiled_model = core.compile_model(model_path, device)

    def __call__(self, idx, past_kv=None):
        if past_kv is None:
            past_kv = self._init_past_kv()
        outs = self.compiled_model([idx, *past_kv], share_outputs=True)
        return self.postprocess_outputs(outs)

    def postprocess_outputs(self, outs):
        net_outs = self.compiled_model.outputs
        logits = outs[net_outs[0]]
        kv_cache = []
        for out_tensor in net_outs[1:]:
            kv_cache.append(outs[out_tensor])
        return logits, kv_cache

    def _init_past_kv(self):
        inputs = []
        for input_t in self.compiled_model.inputs[1:]:
            input_shape = input_t.partial_shape
            input_shape[0] = 1
            input_shape[2] = 0
            inputs.append(ov.Tensor(ov.Type.f32, input_shape.get_shape()))
        return inputs


class OVBarkFineEncoder:
    def __init__(self, core, device, model_dir, num_lm_heads=7):
        self.feats_compiled_model = core.compile_model(os.path.join(model_dir,"bark_fine_feature_extractor.xml"), device)
        self.feats_out = self.feats_compiled_model.output(0)
        lm_heads = []
        for i in range(num_lm_heads):
            lm_heads.append(core.compile_model(os.path.join(model_dir, f"bark_fine_lm_{i}.xml"), device))
        self.lm_heads = lm_heads

    def __call__(self, pred_idx, idx):
        feats = self.feats_compiled_model([ov.Tensor(pred_idx), ov.Tensor(idx)])[self.feats_out]
        lm_id = pred_idx - 1
        logits = self.lm_heads[int(lm_id)](feats)[0]
        return logits