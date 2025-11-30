class Card:
    """Class representing a playing card"""
    
    def __init__(self, suit, rank):
        """
        Initialize a card with suit and rank
        
        Args:
            suit (str): The suit of the card (Hearts, Diamonds, Clubs, Spades)
            rank (str): The rank of the card (2-10, J, Q, K, A)
        """
        self.suit = suit
        self.rank = rank
        self.suits_symbols = {
            'Hearts': '♥',
            'Diamonds': '♦',
            'Clubs': '♣',
            'Spades': '♠'
        }
        self.values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 10, 'Q': 10, 'K': 10, 'A': 11
        }
    
    def get_value(self):
        """Return the numeric value of the card"""
        return self.values[self.rank]
    
    def is_ace(self):
        """Check if card is an Ace"""
        return self.rank == 'A'
    
    def __str__(self):
        """String representation of the card"""
        return f"{self.rank} of {self.suit}"
    
    def get_ascii_art(self, hidden=False):
        """
        Return ASCII art representation of the card
        
        Args:
            hidden (bool): If True, show card back instead of face
            
        Returns:
            list: List of strings representing each line of the card
        """
        if hidden:
            return [
                "┌─────────┐",
                "│░░░░░░░░░│",
                "│░░░░░░░░░│",
                "│░░░░░░░░░│",
                "│░░░░░░░░░│",
                "│░░░░░░░░░│",
                "└─────────┘"
            ]
        
        suit_symbol = self.suits_symbols[self.suit]
        rank_display = self.rank.ljust(2)
        
        # Color codes for terminal
        if self.suit in ['Hearts', 'Diamonds']:
            color = '\033[91m'  # Red
        else:
            color = '\033[90m'  # Dark gray/black
        reset = '\033[0m'
        
        return [
            "┌─────────┐",
            f"│{color}{rank_display}{reset}       │",
            f"│  {color}{suit_symbol}{reset}      │",
            f"│    {color}{suit_symbol}{reset}    │",
            f"│      {color}{suit_symbol}{reset}  │",
            f"│       {color}{rank_display}{reset}│",
            "└─────────┘"
        ]
    
    def __repr__(self):
        """Developer-friendly representation"""
        return f"Card('{self.suit}', '{self.rank}')"