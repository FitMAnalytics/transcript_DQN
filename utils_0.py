import random

def generate_customer_profile():
    '''
    Generate a customer profile based on the customer's profile
    '''
    return {
        "patience": random.uniform(0.5, 1),
        "determination": random.uniform(0.5, 1)
    }

def raise_new_objection(chat_session, objection_type, agent_msg):
    """Triggers the customer to start a new complaint."""
    instruction = f"""
    [NEW CONCERN]
    The agent just said: "{agent_msg}"
    You have a new concern regarding {objection_type}. 
    Express this concern naturally based on your patience and determination.
    """
    response = chat_session.send_message(instruction)
    return response.text

def react_to_handling(chat_session, logic_result, agent_msg):
    """Renders the reaction to an agent's attempt to solve a problem."""
    instruction = f"""
    [REACTION]
    The agent attempted to address your concern by saying: "{agent_msg}"
    LOGICAL OUTCOME: {logic_result}
    
    If 'Still Skeptical', push back or reiterate the worry. 
    If 'Resolved', acknowledge the point (but don't necessarily buy yet).
    If 'Incentive Rejected', express that the offer doesn't change your mind.
    """
    response = chat_session.send_message(instruction)
    return response.text

def select_next_objection(A_priority, B_priority, C_priority, concern_used):
    priorities = [
        ('A', A_priority, concern_used['A']),
        ('B', B_priority, concern_used['B']),
        ('C', C_priority, concern_used['C'])
    ]
    priorities.sort(key=lambda x: x[1], reverse=True)
    for label, _, used in priorities:
        if not used:
            return f"Objection_{label}"
    return None

def set_objection_indicator(objection_action, concern_used):
    label = objection_action.split("_")[1]
    concern_used[label] = True

def generate_customer_sys_prompt(patience, determination):
    return f"""You are a customer shopping for boots. 
    Your goal is NOT to make decisions, but to SPEAK based on the ACTION and CHARACTERISTICS provided.

    ### YOUR CHARACTERISTICS
    - PATIENCE: {patience} (0.5-1.0). If low, be blunt, short, and irritable. If high, be polite and detailed.
    - DETERMINATION: {determination} (0.5-1.0). If high, sound very skeptical, dismissive, and hard-to-impress. If low, sound open and curious.

    ### YOUR ROLE
    You will be given a CUSTOMER ACTION and a LOGICAL OUTCOME. 
    You must generate natural dialogue that reflects these. 

    ### LOGICAL OUTCOMES TO RENDER:
    - "Resolved": You accept the agent's point. Acknowledge it and move on.
    - "Still Skeptical": You are not convinced. Reiterate your concern or push back.
    - "Purchase": You are happy with the deal and agree to buy.
    - "Leave": You are done and want to end the call.

    ### YOUR OBJECTIONS:
    - Objection_A: Quality (Waterproofing, durability).
    - Objection_B: Price (Budget, competitors).
    - Objection_C: Logistics (Shipping speed, returns).

    CONSTRAINTS: Output ONLY dialogue. Do not explain your reasoning.
    """

