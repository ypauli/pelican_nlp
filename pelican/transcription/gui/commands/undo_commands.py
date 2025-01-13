from PyQt5.QtWidgets import QUndoCommand

class EditWordCommand(QUndoCommand):
    def __init__(self, main_window, idx, old_word, new_word, description="Edit Word"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.old_word = old_word
        self.new_word = new_word

    def redo(self):
        if not self.main_window.controller:
            return
        word_data = self.main_window.controller.get_word_data(self.idx)
        if word_data:
            word_data['word'] = self.new_word
            self.main_window.canvas.words[self.idx]['word'] = self.new_word
            self.main_window.canvas.update_connecting_line(self.idx)

    def undo(self):
        if not self.main_window.controller:
            return
        word_data = self.main_window.controller.get_word_data(self.idx)
        if word_data:
            word_data['word'] = self.old_word
            self.main_window.canvas.words[self.idx]['word'] = self.old_word
            self.main_window.canvas.update_connecting_line(self.idx)


class EditSpeakerCommand(QUndoCommand):
    def __init__(self, main_window, idx, old_speaker, new_speaker, description="Edit Speaker"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.old_speaker = old_speaker
        self.new_speaker = new_speaker

    def redo(self):
        if not self.main_window.controller:
            return
        word_data = self.main_window.controller.get_word_data(self.idx)
        if word_data:
            word_data['speaker'] = self.new_speaker
            self.main_window.canvas.words[self.idx]['speaker'] = self.new_speaker
            self.main_window.canvas.update_connecting_line(self.idx)

    def undo(self):
        if not self.main_window.controller:
            return
        word_data = self.main_window.controller.get_word_data(self.idx)
        if word_data:
            word_data['speaker'] = self.old_speaker
            self.main_window.canvas.words[self.idx]['speaker'] = self.old_speaker
            self.main_window.canvas.update_connecting_line(self.idx)


class MoveBoundaryCommand(QUndoCommand):
    def __init__(self, main_window, idx, boundary_type, old_pos, new_pos, description="Move Boundary"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.boundary_type = boundary_type
        self.old_pos = old_pos
        self.new_pos = new_pos

    def redo(self):
        if not self.main_window.controller:
            return
        if self.boundary_type == 'start':
            self.main_window.controller.update_word_boundaries(self.idx, start_time=self.new_pos)
        else:
            self.main_window.controller.update_word_boundaries(self.idx, end_time=self.new_pos)
        self.main_window.canvas.update_line_position(self.idx, self.boundary_type, self.new_pos)

    def undo(self):
        if not self.main_window.controller:
            return
        if self.boundary_type == 'start':
            self.main_window.controller.update_word_boundaries(self.idx, start_time=self.old_pos)
        else:
            self.main_window.controller.update_word_boundaries(self.idx, end_time=self.old_pos)
        self.main_window.canvas.update_line_position(self.idx, self.boundary_type, self.old_pos) 

class SplitWordCommand(QUndoCommand):
    """Command for splitting a word into two at a specific time point."""
    
    def __init__(self, main_window, idx, split_time, new_word_text, description="Split Word"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.split_time = split_time
        self.new_word_text = new_word_text
        self.old_word = None
        self.new_word = None
        
    def redo(self):
        if not self.main_window.controller:
            return
            
        word = self.main_window.controller.get_word_data(self.idx)
        if not word:
            return
            
        # Store the original word for undo
        if not self.old_word:
            self.old_word = dict(word)
            
        # Create new word data
        if not self.new_word:
            self.new_word = dict(word)
            self.new_word['start_time'] = self.split_time
            word['end_time'] = self.split_time
            self.new_word['word'] = self.new_word_text
            
        # Update the data
        self.main_window.transcript.combined_data.insert(self.idx + 1, self.new_word)
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)
        
    def undo(self):
        if not self.main_window.controller or not self.old_word:
            return
            
        # Remove the new word and restore the original
        self.main_window.transcript.combined_data.pop(self.idx + 1)
        self.main_window.transcript.combined_data[self.idx] = dict(self.old_word)
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)

class AddWordCommand(QUndoCommand):
    """Command for adding a new word at a specific time point."""
    
    def __init__(self, main_window, time_point, word_text, duration=0.1, description="Add Word"):
        super().__init__(description)
        self.main_window = main_window
        self.time_point = time_point
        self.duration = duration
        self.word_text = word_text
        self.word_data = None
        
    def redo(self):
        if not self.main_window.controller:
            return
            
        if not self.word_data:
            # Create new word data
            self.word_data = {
                'word': self.word_text,
                'start_time': self.time_point - self.duration/2,  # Center the word at click point
                'end_time': self.time_point + self.duration/2,
                'speaker': '',  # Empty speaker
                'confidence': 1.0
            }
            
        # Find the correct position to insert the word
        words = self.main_window.transcript.combined_data
        insert_idx = 0
        for i, word in enumerate(words):
            if float(word['start_time']) > self.time_point:
                insert_idx = i
                break
            insert_idx = i + 1
            
        # Insert the word and refresh display
        self.main_window.transcript.combined_data.insert(insert_idx, self.word_data)
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)
        
    def undo(self):
        if not self.main_window.controller or not self.word_data:
            return
            
        # Find and remove the word
        words = self.main_window.transcript.combined_data
        for i, word in enumerate(words):
            if (word['start_time'] == self.word_data['start_time'] and 
                word['end_time'] == self.word_data['end_time']):
                self.main_window.transcript.combined_data.pop(i)
                break
                
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data) 

class DeleteWordCommand(QUndoCommand):
    """Command for deleting a word."""
    
    def __init__(self, main_window, idx, description="Delete Word"):
        super().__init__(description)
        self.main_window = main_window
        self.idx = idx
        self.deleted_word = None
        
    def redo(self):
        if not self.main_window.controller:
            return
            
        # Store the word for undo if not already stored
        if not self.deleted_word:
            self.deleted_word = dict(self.main_window.transcript.combined_data[self.idx])
            
        # Remove the word
        self.main_window.transcript.combined_data.pop(self.idx)
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)
        
    def undo(self):
        if not self.main_window.controller or not self.deleted_word:
            return
            
        # Reinsert the word
        self.main_window.transcript.combined_data.insert(self.idx, dict(self.deleted_word))
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)

class BulkEditCommand(QUndoCommand):
    """Command for bulk editing multiple words."""
    
    def __init__(self, main_window, indices, changes, description="Bulk Edit"):
        super().__init__(description)
        self.main_window = main_window
        self.indices = indices
        self.changes = changes  # Dict of field:new_value pairs
        self.old_values = {}  # Will store original values for undo
        
    def redo(self):
        if not self.main_window.controller:
            return
            
        # Store original values if not already stored
        if not self.old_values:
            for idx in self.indices:
                word = self.main_window.transcript.combined_data[idx]
                self.old_values[idx] = {
                    field: word.get(field, '') 
                    for field in self.changes.keys()
                }
        
        # Apply changes
        for idx in self.indices:
            word = self.main_window.transcript.combined_data[idx]
            for field, value in self.changes.items():
                word[field] = value
                
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)
        
    def undo(self):
        if not self.main_window.controller:
            return
            
        # Restore original values
        for idx, old_values in self.old_values.items():
            word = self.main_window.transcript.combined_data[idx]
            for field, value in old_values.items():
                word[field] = value
                
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)

class BulkDeleteCommand(QUndoCommand):
    """Command for deleting multiple words."""
    
    def __init__(self, main_window, indices, description="Bulk Delete"):
        super().__init__(description)
        self.main_window = main_window
        self.indices = sorted(indices, reverse=True)  # Sort in reverse to delete from end
        self.deleted_words = {}
        
    def redo(self):
        if not self.main_window.controller:
            return
            
        # Store words for undo if not already stored
        if not self.deleted_words:
            for idx in self.indices:
                self.deleted_words[idx] = dict(self.main_window.transcript.combined_data[idx])
        
        # Remove words from end to start
        for idx in self.indices:
            self.main_window.transcript.combined_data.pop(idx)
            
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data)
        
    def undo(self):
        if not self.main_window.controller:
            return
            
        # Reinsert words from start to end
        for idx in reversed(self.indices):
            self.main_window.transcript.combined_data.insert(idx, dict(self.deleted_words[idx]))
            
        self.main_window.canvas.load_words(self.main_window.transcript.combined_data) 