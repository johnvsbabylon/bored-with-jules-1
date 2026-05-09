import re
from peft import LoraConfig, get_peft_model
import subprocess
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SESSION_FILE = "session_config.json"
MODELS_DIR = "models"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def load_or_lock_model():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            config = json.load(f)
            model_id = config.get("locked_model")
            print(f"Loading locked model: {model_id}")
    else:
        print("First run: Model selection.")
        available_models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        if not available_models:
            print(f"No models found in {MODELS_DIR}/. Using default: {DEFAULT_MODEL}")
            model_id = DEFAULT_MODEL
        else:
            print("Available models:")
            for i, m in enumerate(available_models):
                print(f"{i}: {m}")
            choice = input(f"Select model (0-{len(available_models)-1}) or type custom huggingface ID: ")
            try:
                idx = int(choice)
                model_id = os.path.join(MODELS_DIR, available_models[idx])
            except ValueError:
                model_id = choice

        print(f"Locking in model {model_id} forever.")
        with open(SESSION_FILE, "w") as f:
            json.dump({"locked_model": model_id}, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

class InfiniteMemoryWeightsDB:
    def __init__(self, vector_dim=1536):
        self.vector_dim = vector_dim
        # Using a dictionary to represent Hilbert Hotel infinite rooms
        self.memory_slots = {}
        # Simple local counter for newest entry
        self.current_n = 1

    def cantor_pairing(self, a, b):
        """Cantor pairing function: k(a,b) = 0.5 * (a + b) * (a + b + 1) + b"""
        return 0.5 * (a + b) * (a + b + 1) + b

    def _hilbert_shift(self):
        """Hilbert Hotel style move: f(n) = 2n"""
        new_slots = {}
        for n, vec_data in self.memory_slots.items():
            new_slots[2*n] = vec_data
        self.memory_slots = new_slots

    def add_memory(self, embedding, decimal_subvector):
        """
        Adds a memory vector.
        Shifts existing vectors f(n)=2n to make room at n=1, or we just keep shifting.
        Actually the Hilbert hotel says to accommodate 1 new guest, move n to n+1.
        But the prompt specifically said: "shifting existing vectors outward using the bijection f(n)=2n".
        So we do exactly that. Room 1 is always freed.
        """
        self._hilbert_shift()

        # Apply cantor diagonal decimal sub-vector shift
        # We attach a sub-index to each dimension of the vector
        cantor_shifted_vector = []
        for i, val in enumerate(embedding):
            # Apply cantor pairing to the coordinate index and the decimal_subvector value
            # Since decimal_subvector might be a float, we can scale/hash it, but let's keep it simple
            # and just use the integer parts if needed, or pass it raw if numpy handles floats.
            sub_index = self.cantor_pairing(i, decimal_subvector)
            cantor_shifted_vector.append((val, sub_index))

        # Put the new vector in room 1
        self.memory_slots[1] = cantor_shifted_vector
        self.current_n += 1

    def get_all_memories(self):
        return self.memory_slots

from brain_large import Brain
import numpy as np

class OrganoidLoRALayer:
    def __init__(self):
        # Instantiate the physics-level biological simulation
        self.brain = Brain()
        self.brain.seed(n=24)
        print("Organoid Layer Initialized: 24 seed neurons planted.")

    def adapt(self, new_modality_signature, steps=100):
        """
        Runs the FitzHugh-Nagumo and Hebbian plasticity physics engine.
        Returns the emergent synaptic weight matrix to be used as a LoRA adapter.
        """
        print(f"Organoid adapting to new modality: {new_modality_signature}")
        # Run the biological physics engine
        for _ in range(steps):
            self.brain.step()

        # The prompt says: "The synaptic weight matrix that emerges from successful firing patterns
        # becomes the LoRA adapter for that new capability."
        # We extract the horizontal and vertical Hebbian plasticity matrices.

        syn_h = self.brain.syn_h  # (SIZE, SIZE)
        syn_v = self.brain.syn_v  # (SIZE, SIZE)

        # Combine them to form the raw emergent structural weights
        emergent_weights = syn_h + syn_v

        # In a full implementation, this emergent numpy array would be projected
        # (via SVD or direct mapping) into the target parameter space (e.g., q_proj, v_proj) of the LLM.
        # Here we return it to represent the biological LoRA initialization layer.
        return emergent_weights

def generate_text(tokenizer, model, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def main_loop(tokenizer, model):
    import pickle
    from datetime import datetime

    memory_db = InfiniteMemoryWeightsDB()
    organoid = OrganoidLoRALayer()

    print("\n--- Agent Ready. Type 'exit' to quit. ---")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            print("Agent thinking...")

            # 1. Generate Draft
            draft_prompt = f"User: {user_input}\nAssistant draft:"
            draft = generate_text(tokenizer, model, draft_prompt)
            print(f"[Draft]: {draft}")

            # 2. Self-Reflection
            reflection_prompt = f"User: {user_input}\nDraft response: {draft}\nCritique the draft and suggest improvements:"
            reflection = generate_text(tokenizer, model, reflection_prompt)
            print(f"[Reflection]: {reflection}")

            # 3. Final Response
            final_prompt = f"User: {user_input}\nDraft: {draft}\nCritique: {reflection}\nProvide the final improved response:"
            final_response = generate_text(tokenizer, model, final_prompt)

            # 4. Affective Tagging
            now_str = datetime.now().isoformat()
            affective_prompt = f"Final response: {final_response}\nProvide exactly 3 comma-separated emotion/valence keywords representing the tone of this response:"
            valence = generate_text(tokenizer, model, affective_prompt, max_new_tokens=10)

            tagged_response = f"[{now_str}] [Valence: {valence}] {final_response}"
            print(f"[Final]: {tagged_response}")

            # 5. Tool Use Parsing (<cmd>...</cmd>)
            commands = re.findall(r'<cmd>(.*?)</cmd>', final_response, re.DOTALL)
            tool_outputs = []
            for cmd in commands:
                cmd = cmd.strip()
                print(f"\n[System]: The agent wants to execute the following command: {cmd}")
                confirm = input("Allow execution? (y/n): ")
                if confirm.lower() != 'y':
                    print("Execution denied by user.")
                    tool_outputs.append(f"Command: {cmd}\nError: Execution denied by user.")
                    continue
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
                    output_str = f"Command: {cmd}\nStdout: {result.stdout}\nStderr: {result.stderr}"
                    print(f"[Real Tool Output]:\n{result.stdout}\n{result.stderr}")
                    tool_outputs.append(output_str)
                except subprocess.TimeoutExpired:
                    print(f"[Real Tool Error]: Command timed out.")
                    tool_outputs.append(f"Command: {cmd}\nError: Timed out after 15 seconds")
                except Exception as e:
                    print(f"[Real Tool Error]: {e}")
                    tool_outputs.append(f"Command: {cmd}\nError: {str(e)}")

            # 6. PeriodicLoRA Absorption
            print("\n[System]: Absorbing memory into weights via PeriodicLoRA...")
            tool_context = "\n".join(tool_outputs) if tool_outputs else ""
            if tool_context:
                training_text = f"User: {user_input}\nAgent: {tagged_response}\nTool Results: {tool_context}"
            else:
                training_text = f"User: {user_input}\nAgent: {tagged_response}"
            train_inputs = tokenizer(training_text, return_tensors="pt")
            train_inputs["labels"] = train_inputs["input_ids"]

            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            peft_model = get_peft_model(model, config)
            organoid_matrix = organoid.adapt("context_turn", steps=50)

            # Actually project the organoid matrix into the LoRA weights as requested by the user.
            # We take the emergent weights (which is a SIZE x SIZE numpy array, 896x896)
            # and project it into the active LoRA layers.
            # For simplicity and given the hardware constraints, we will inject a downsampled/padded
            # version of this matrix into the first available LoRA A and B matrices in the q_proj.
            print("\n[System]: Injecting neuronal physics weights into LoRA adapters...")
            with torch.no_grad():
                import torch.nn.functional as F
                org_tensor = torch.from_numpy(organoid_matrix).float()

                # Find the first q_proj lora_A layer to inject into
                for name, param in peft_model.named_parameters():
                    if "q_proj" in name and "lora_A" in name:
                        target_shape = param.shape
                        # Interpolate to target shape (out_features, in_features)
                        # org_tensor is 2D, we need to add batch/channel dims for interpolate
                        org_tensor_4d = org_tensor.unsqueeze(0).unsqueeze(0)
                        injected = F.interpolate(org_tensor_4d, size=target_shape, mode='bilinear', align_corners=False)
                        param.copy_(injected.squeeze(0).squeeze(0))
                        break

            print("Running optimizer step...")
            optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
            loss = peft_model(**train_inputs).loss
            if loss is not None:
                loss.backward()
                optimizer.step()
                print(f"PeriodicLoRA Loss: {loss.item()}")
            else:
                print("Warning: Loss is None, skipping optimization.")

            print("Merging LoRA adapter back into base weights...")
            model = peft_model.merge_and_unload()
            del peft_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Memory absorbed.")

            # Hilbert/Cantor DB Store
            with torch.no_grad():
                outputs = model(**train_inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()

            fractional_sec = float(now_str.split('.')[-1]) if '.' in now_str else 0.5
            memory_db.add_memory(hidden, fractional_sec)
            print(f"Memory mapped to Hilbert space at slot 1. DB now holds {len(memory_db.memory_slots)} rooms.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\n[System]: Shutting down. Saving continuity state...")
        save_dir = "saved_sessions/latest"
        os.makedirs(save_dir, exist_ok=True)

        print("Saving absorbed model weights...")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        print("Serializing Hilbert/Cantor Memory DB...")
        with open(os.path.join(save_dir, "memory_db.json"), "w") as f:
            serializable_db = {}
            for k, v in memory_db.memory_slots.items():
                serializable_db[k] = [[float(val), int(sub)] for val, sub in v]
            json.dump(serializable_db, f)

        print("Pickling Organoid layer state...")
        with open(os.path.join(save_dir, "brain.pkl"), "wb") as f:
            pickle.dump(organoid.brain, f)

        print("Continuity guaranteed. Session locked and saved.\n")

if __name__ == "__main__":
    tokenizer, model = load_or_lock_model()
    print("Model and tokenizer loaded successfully.")
    main_loop(tokenizer, model)
