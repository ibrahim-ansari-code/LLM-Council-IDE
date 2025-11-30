import random
from card import Card
import os

class Deck:
    """Class representing a deck of cards"""
    
    def __init__(self):
        """Initialize a standard 52-card deck"""
        self.cards = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))
        
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck"""
        random.shuffle(self.cards)
    
    def deal_card(self):
        """Deal one card from the deck"""
        if len(self.cards) == 0:
            # Reshuffle if deck is empty
            self.__init__()
        return self.cards.pop()


class Hand:
    """Class representing a hand of cards"""
    
    def __init__(self):
        """Initialize an empty hand"""
        self.cards = []
    
    def add_card(self, card):
        """Add a card to the hand"""
        self.cards.append(card)
    
    def get_value(self):
        """Calculate the value of the hand, accounting for Aces"""
        value = 0
        aces = 0
        
        for card in self.cards:
            value += card.get_value()
            if card.is_ace():
                aces += 1
        
        # Adjust for Aces (count as 1 instead of 11 if needed)
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        return value
    
    def is_blackjack(self):
        """Check if hand is a blackjack (21 with 2 cards)"""
        return len(self.cards) == 2 and self.get_value() == 21
    
    def is_bust(self):
        """Check if hand is bust (over 21)"""
        return self.get_value() > 21
    
    def display(self, hide_first=False):
        """Display all cards in the hand as ASCII art"""
        if not self.cards:
            return
        
        # Get ASCII art for each card
        card_arts = []
        for i, card in enumerate(self.cards):
            hidden = (i == 0 and hide_first)
            card_arts.append(card.get_ascii_art(hidden))
        
        # Print cards side by side
        for line_idx in range(7):
            line = ""
            for card_art in card_arts:
                line += card_art[line_idx] + "  "
            print(line)


class PlayerStats:
    """Class to manage player statistics"""
    
    def __init__(self, username):
        """Initialize player stats"""
        self.username = username
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        self.balance = 1000
    
    def to_string(self):
        """Convert stats to string for file storage"""
        return f"{self.username},{self.wins},{self.losses},{self.pushes},{self.balance}\n"
    
    @staticmethod
    def from_string(line):
        """Create PlayerStats from file line"""
        try:
            parts = line.strip().split(',')
            if len(parts) == 5:
                stats = PlayerStats(parts[0].strip())
                stats.wins = int(parts[1].strip())
                stats.losses = int(parts[2].strip())
                stats.pushes = int(parts[3].strip())
                stats.balance = int(parts[4].strip())
                return stats
        except (ValueError, IndexError):
            # Skip corrupted lines
            return None
        return None


class BlackjackGame:
    """Main Blackjack game class"""
    
    def __init__(self):
        """Initialize the game"""
        self.player_file = "players.txt"
        self.current_player = None
        self.deck = None
        self.player_hand = None
        self.dealer_hand = None
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def load_or_create_player(self, username):
        """Load player from file or create new player"""
        # Create file if it doesn't exist
        if not os.path.exists(self.player_file):
            with open(self.player_file, 'w') as f:
                pass
        
        # Search for player in file
        try:
            with open(self.player_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        stats = PlayerStats.from_string(line)
                        if stats and stats.username == username:
                            return stats
        except Exception as e:
            print(f"Warning: Error reading player file: {e}")
            print("Creating new player profile...")
        
        # Player not found, create new
        return PlayerStats(username)
    
    def save_player(self):
        """Save current player stats to file"""
        if not self.current_player:
            return
        
        # Read all players
        players = []
        if os.path.exists(self.player_file):
            try:
                with open(self.player_file, 'r') as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            stats = PlayerStats.from_string(line)
                            if stats:
                                players.append(stats)
            except Exception as e:
                print(f"Warning: Error reading player file: {e}")
        
        # Update or add current player
        found = False
        for i, player in enumerate(players):
            if player.username == self.current_player.username:
                players[i] = self.current_player
                found = True
                break
        
        if not found:
            players.append(self.current_player)
        
        # Write all players back
        try:
            with open(self.player_file, 'w') as f:
                for player in players:
                    f.write(player.to_string())
        except Exception as e:
            print(f"Warning: Error saving player file: {e}")
    
    def login(self):
        """Handle player login"""
        print("=" * 50)
        print("WELCOME TO BLACKJACK".center(50))
        print("=" * 50)
        print()
        username = input("Enter your username: ").strip()
        
        if not username:
            print("Invalid username!")
            return False
        
        self.current_player = self.load_or_create_player(username)
        print(f"\nWelcome, {self.current_player.username}!")
        print(f"Balance: ${self.current_player.balance}")
        print(f"Record: {self.current_player.wins}W - {self.current_player.losses}L - {self.current_player.pushes}P")
        input("\nPress Enter to continue...")
        return True
    
    def display_game_state(self, hide_dealer_card=True):
        """Display current game state"""
        self.clear_screen()
        print("=" * 50)
        print("BLACKJACK".center(50))
        print("=" * 50)
        print(f"\nPlayer: {self.current_player.username} | Balance: ${self.current_player.balance}")
        print("\n" + "DEALER'S HAND".center(50))
        if hide_dealer_card:
            visible_value = self.dealer_hand.cards[1].get_value() if len(self.dealer_hand.cards) > 1 else 0
            print(f"Showing: {visible_value}")
        else:
            print(f"Value: {self.dealer_hand.get_value()}")
        self.dealer_hand.display(hide_first=hide_dealer_card)
        
        print("\n" + "YOUR HAND".center(50))
        print(f"Value: {self.player_hand.get_value()}")
        self.player_hand.display()
        print()
    
    def place_bet(self):
        """Get bet amount from player"""
        while True:
            try:
                bet = int(input(f"\nPlace your bet (Balance: ${self.current_player.balance}): $"))
                if bet <= 0:
                    print("Bet must be positive!")
                elif bet > self.current_player.balance:
                    print("Insufficient balance!")
                else:
                    return bet
            except ValueError:
                print("Invalid input! Enter a number.")
    
    def deal_initial_cards(self):
        """Deal initial two cards to player and dealer"""
        self.player_hand.add_card(self.deck.deal_card())
        self.dealer_hand.add_card(self.deck.deal_card())
        self.player_hand.add_card(self.deck.deal_card())
        self.dealer_hand.add_card(self.deck.deal_card())
    
    def player_turn(self):
        """Handle player's turn"""
        while True:
            self.display_game_state(hide_dealer_card=True)
            
            if self.player_hand.is_bust():
                print("BUST! You lose!")
                input("\nPress Enter to continue...")
                return False
            
            if self.player_hand.is_blackjack():
                print("BLACKJACK!")
                input("\nPress Enter to continue...")
                return True
            
            choice = input("(H)it or (S)tand? ").strip().upper()
            
            if choice == 'H':
                self.player_hand.add_card(self.deck.deal_card())
            elif choice == 'S':
                return True
            else:
                print("Invalid choice! Enter H or S.")
                input("Press Enter to continue...")
    
    def dealer_turn(self):
        """Handle dealer's turn (dealer must hit on 16 or less, stand on 17+)"""
        while self.dealer_hand.get_value() < 17:
            self.dealer_hand.add_card(self.deck.deal_card())
    
    def determine_winner(self, bet):
        """Determine winner and update stats"""
        self.display_game_state(hide_dealer_card=False)
        
        player_value = self.player_hand.get_value()
        dealer_value = self.dealer_hand.get_value()
        
        print("\n" + "=" * 50)
        
        if self.player_hand.is_bust():
            print("You BUST! Dealer wins!")
            self.current_player.losses += 1
            self.current_player.balance -= bet
        elif self.dealer_hand.is_bust():
            print("Dealer BUSTS! You win!")
            self.current_player.wins += 1
            self.current_player.balance += bet
        elif self.player_hand.is_blackjack() and not self.dealer_hand.is_blackjack():
            print("BLACKJACK! You win 1.5x your bet!")
            self.current_player.wins += 1
            self.current_player.balance += int(bet * 1.5)
        elif player_value > dealer_value:
            print(f"You win! {player_value} beats {dealer_value}!")
            self.current_player.wins += 1
            self.current_player.balance += bet
        elif player_value < dealer_value:
            print(f"Dealer wins! {dealer_value} beats {player_value}!")
            self.current_player.losses += 1
            self.current_player.balance -= bet
        else:
            print(f"Push! Both have {player_value}.")
            self.current_player.pushes += 1
        
        print("=" * 50)
        print(f"\nNew Balance: ${self.current_player.balance}")
        input("\nPress Enter to continue...")
    
    def play_round(self):
        """Play one round of blackjack"""
        if self.current_player.balance <= 0:
            print("\nYou're out of money! Game Over!")
            return False
        
        # Initialize new round
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        
        # Place bet
        self.clear_screen()
        bet = self.place_bet()
        
        # Deal initial cards
        self.deal_initial_cards()
        
        # Check for immediate blackjack
        if self.player_hand.is_blackjack() and self.dealer_hand.is_blackjack():
            self.display_game_state(hide_dealer_card=False)
            print("\nBoth have BLACKJACK! Push!")
            self.current_player.pushes += 1
            input("\nPress Enter to continue...")
            return True
        
        # Player's turn
        if not self.player_turn():
            self.determine_winner(bet)
            self.save_player()
            return True
        
        # Dealer's turn
        if not self.player_hand.is_bust():
            self.dealer_turn()
        
        # Determine winner
        self.determine_winner(bet)
        self.save_player()
        return True
    
    def run(self):
        """Main game loop"""
        if not self.login():
            return
        
        while True:
            if not self.play_round():
                break
            
            self.clear_screen()
            print(f"\nCurrent Balance: ${self.current_player.balance}")
            print(f"Record: {self.current_player.wins}W - {self.current_player.losses}L - {self.current_player.pushes}P")
            
            play_again = input("\nPlay another round? (Y/N): ").strip().upper()
            if play_again != 'Y':
                break
        
        self.clear_screen()
        print("\n" + "=" * 50)
        print("THANKS FOR PLAYING!".center(50))
        print("=" * 50)
        print(f"\nFinal Stats for {self.current_player.username}:")
        print(f"Balance: ${self.current_player.balance}")
        print(f"Wins: {self.current_player.wins}")
        print(f"Losses: {self.current_player.losses}")
        print(f"Pushes: {self.current_player.pushes}")
        print("\n" + "=" * 50)


if __name__ == "__main__":
    game = BlackjackGame()
    game.run()