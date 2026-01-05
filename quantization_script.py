            from torch.export import export
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner, ConfigPrecisionType
            from executorch.exir import to_edge_transform_and_lower
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            
            # dummy_input_tensor = torch.randint(128, size=(1,), dtype=torch.int32).to("cpu")  
            # dummy_input_tensor = torch.tensor([1], dtype=torch.long).to("cpu")          
            dummy_input_tensor = torch.tensor([1], dtype=torch.long).to("cpu")          
            h = torch.zeros(24, 4096, dtype=torch.float).to("cpu")
            # h = torch.zeros(24, 4096, dtype=torch.half).to("cpu")
            fifo = torch.zeros(24, 4096, 4)
            # fifo = torch.zeros(24, 4096, 4, dtype=torch.half).to("cpu")
            mamba.reset_states()
            
            # Export to executorch
            dummy_inputs = (dummy_input_tensor,  h, fifo)
            partitioner=[XnnpackPartitioner()]
            
            exported_program = export(mamba, dummy_inputs)
            executorch_program = to_edge_transform_and_lower(
                    exported_program,
                    partitioner=partitioner,
                )                 
            executorch_program = executorch_program.to_executorch()
            with open(f"{args.export}.pte", "wb") as file:
                file.write(executorch_program.buffer)
            
            print(f"✅ XNN : Succesfully saved model as {args.export}.pte")            
            
            ### Quantizer and export
            quant = 1
            if quant:
                qparams = get_symmetric_quantization_config(is_per_channel=False) # (1)
                quantizer = XNNPACKQuantizer()
                quantizer.set_global(qparams)
                
                training_ep = torch.export.export_for_training(mamba, dummy_inputs).module() # (2)
                prepared_model = prepare_pt2e(training_ep, quantizer) # (3)



                #for calib_inputs in calib_inputs_list: # Replace with representative model inputs
                calibration_sentences = [
                    "How do I reset the refrigerator?",
                    "The mobile app is not connecting to the device.",
                    "List the steps to sync the smart fridge with WiFi.",
                    "What are the power requirements for this model?"
                ]

                # Combine them into one long sequence or loop through them
                all_tokens = []
                for s in calibration_sentences:
                    tokens = stokenizer(s, return_tensors='pt').input_ids[0]
                    all_tokens.extend(tokens.tolist())

                calib_tokens = torch.tensor(all_tokens)
                with torch.no_grad():
                    print("Calibrating with real data...")
                    # Reset states to ensure we start clean
                    temp_h = torch.zeros(24, 4096, dtype=torch.float)
                    temp_fifo = torch.zeros(24, 4096, 4, dtype=torch.float)
                    
                    # Run at least 20-30 tokens to 'warm up' the state ranges
                    for tkn in calib_tokens[:30]: 
                        prepared_model(tkn.view(1), temp_h, temp_fifo)
                # prepared_model(dummy_input_tensor,  h, fifo) # (4) Calibrate


                
                quantized_model = convert_pt2e(prepared_model) # (5)
                print("Conversion to INT8 complete.")
                final_export = torch.export.export(quantized_model, dummy_inputs)

                for node in exported_program.graph.nodes:
                    if node.op == "placeholder":
                        tm = node.meta["tensor_meta"]
                        print(node.name, tm.shape, tm.dtype)

                executorch_program = to_edge_transform_and_lower(
                    final_export,
                    partitioner = [XnnpackPartitioner(
                        config_precisions=[ConfigPrecisionType.STATIC_QUANT]
                    )],
                ).to_executorch()
                with open(f"{args.export}_int8.pte", "wb") as file:
                    file.write(executorch_program.buffer)
                                
                print(f"✅ Quant XNN: Succesfully saved model as {args.export}.pte")
