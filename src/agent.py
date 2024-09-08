# General Agent playing Pokemon Red 
# A self-evolving Agent backed by VLM and two memory modules
# Two main mechanisms:
# 1. VLM-based self-evolving heuristics (state - action - next_state like plans)
# 2. Reflex-based action sequence models (multiple basic transformer models or one transformer model with different heads)
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from red_gym_env import PokeRedEnv


heuristic_map = {
    "DOWN": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "UP": 3,
    "A": 4,
    "B": 5,
    "START": 6
}


class AGI_Agent:
    def __init__(self, action_models, memory_size=1000, gb_path=None, init_state=None):
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-08-26"
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.action_models = action_models  # List of Transformer models for action sequences
        self.heuristic_memory = []  # List to store heuristics
        self.action_memory = []  # List to store action sequences
        self.memory_size = memory_size  # Maximum size of memory modules
        
        # Initialize the PokeRedEnv
        self.env = PokeRedEnv(gb_path, init_state) if gb_path and init_state else None
        
    def generate_heuristic(self, state):
        # Generate a new heuristic based on the current state
        # Convert numpy array to PIL Image
        state_image = Image.fromarray(state['screen'])
        enc_image = self.vlm_model.encode_image(state_image)
        
        
        available_action_prompt = "Available actions to control the game:\n" \
            "UP: Move upwards\n" \
            "DOWN: Move downwards\n" \
            "LEFT: Move to the left\n" \
            "RIGHT: Move to the right\n" \
            "A: Interact or confirm\n" \
            "B: Cancel or go back\n" \
            "START: Open the menu or pause the game"
        heuristic_prompt = f"Analyze the current game state in Pokemon Red. Describe what you see, including the player's location, nearby objects or characters, and any visible menu or dialog. Based on this analysis, suggest the most appropriate action to progress in the game. Your response should be structured as follows:\n\n" \
            "1. Game State Description: [Your detailed description]\n" \
            "2. Recommended Action: [ONE of the available actions]\n" \
            "3. Reasoning: [Brief explanation for the recommended action]\n\n" \
            f"{available_action_prompt}\n\n" \
            "Ensure your recommended action is ONE of the listed available actions."
        
        heuristic = self.vlm_model.answer_question(enc_image, heuristic_prompt, self.tokenizer)
        
        print("VLM Heuristic: ", heuristic)
        
        return heuristic
    
    def select_action(self, state):
        # Select an action based on heuristics and action model
        heuristic = self.generate_heuristic(state)
        
        action_idx = self.env.action_space.sample()
        for action in heuristic_map:
            if action in heuristic:
                action_idx = heuristic_map[action]
        
        return action_idx
    
    def update_memories(self, state, action, next_state, reward, done):
        # Update both heuristic and action memories
        heuristic = self.generate_heuristic(state)
        self.heuristic_memory.append((state, heuristic, action, next_state, reward, done))
        self.action_memory.append((state, action, next_state, reward, done))
        
        # Trim memories if they exceed the maximum size
        if len(self.heuristic_memory) > self.memory_size:
            self.heuristic_memory = self.heuristic_memory[-self.memory_size:]
        if len(self.action_memory) > self.memory_size:
            self.action_memory = self.action_memory[-self.memory_size:]
    
    def evolve_heuristics(self):
        # Evolve and refine existing heuristics
        # This could involve analyzing the heuristic_memory to identify successful patterns
        pass
    
    def train_action_model(self):
        # Train the action model based on successful action sequences
        # This should use the action_memory to update the action_models
        pass

    def play_episode(self):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.update_memories(state, action, next_state, reward, done)
            
            state = next_state
            total_reward += reward
        
        return total_reward
    
    
    
    
# Pokemon Agent

agent = AGI_Agent([], gb_path="Pokemon Red.gb", init_state="init.state")
agent.play_episode()