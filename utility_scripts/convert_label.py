
LABEL_NROMAL = 0
LABEL_FALL = 1

# 1st order event
EVENT_NORMAL2FALL = "EVENT_NORMAL2FALL"
EVENT_FALL2NORMAL = "EVENT_FALL2NORMAL"

# # 2nd order event
# EVENT_NORMAL2FALL2NORMAL = 2
# EVENT_FALL2NORMAL2FALL = 3

XI_FNF = 30 # the time priod to determine whether to convert the event, XI < real_action_time_period XI = 100/<spilt>
XI_NFN = 15 # need to change
XI_NFN_PREV = 3

result_label = [1,0,1,1,0,1]
events = [] # an event: (EVENT_TYPE, INDEX_OF_LABEL), index of label is the cdr index of an event
# events_2nd = []

def main():
    # initialize the covertion result
    converted_result_label =  result_label
    
    # build up events
    events = build_events(converted_result_label)
    
    # convert FALL to NORMAL, filtering the sigular points
    for i in range(len(events)):
        if i > 0:
            delta_event = events[i][1] - events[i-1][1]
            if(events[i-1][0] == EVENT_NORMAL2FALL): # case N2F2N
                if(delta_event < XI_NFN_PREV):
                    converted_result_label[events[i-1][1]:events[i][1]] = [LABEL_NROMAL] * delta_event
    
    # re-build the events according to the new converted_result_label
    events.clear()
    events = build_events(converted_result_label)
    
    # convert NORMAL to FALL
    for i in range(len(events)):
        if i > 0:
            delta_event = events[i][1] - events[i-1][1]
            if(events[i-1][0] == EVENT_FALL2NORMAL): # case F2N2F
                if(delta_event < XI_FNF):
                    converted_result_label[events[i-1][1]:events[i][1]] = [LABEL_FALL] * delta_event
    
    print("DEBUG: ", events, converted_result_label)
    
    # re-build the events according to the new converted_result_label
    events.clear()
    events = build_events(converted_result_label)
    
    # convert FALL to NORMAL
    for i in range(len(events)):
        if i > 0:
            delta_event = events[i][1] - events[i-1][1]
            if(events[i-1][0] == EVENT_NORMAL2FALL): # case N2F2N
                if(delta_event < XI_NFN):
                    converted_result_label[events[i-1][1]:events[i][1]] = [LABEL_NROMAL] * delta_event

    
    print("RESULT: ", events, converted_result_label)

def build_events(result_list):
    # build up events
    events_local = []
    for i in range(len(result_list)):
        if i > 0:
            if(result_list[i] == LABEL_NROMAL and result_list[i-1]==LABEL_FALL):
                events_local.append((EVENT_FALL2NORMAL, i))
            elif(result_list[i] == LABEL_FALL and result_list[i-1]==LABEL_NROMAL):
                events_local.append((EVENT_NORMAL2FALL, i))
    return events_local

if __name__ == "__main__":
    main()