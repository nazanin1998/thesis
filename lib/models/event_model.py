class EventModel:
    def __init__(self, path, rumors, non_rumors):
        self.name = path
        self.rumors = rumors
        self.non_rumors = non_rumors

    def to_string(self, index=''):
        return f"\t\tEvent {index} ==> {self.get_event_title()} \t==>" \
               f"\trumours: {self.get_rumour_source_tweets_counts()}," \
               f"\tnon_rumours: {self.get_non_rumour_source_tweets_counts()}" \
               f"\trumours_all: {self.get_rumour_total_tweets_counts()}" \
               f"\tnon_rumours_all: {self.get_non_rumour_total_tweets_counts()}"

    def to_table_array(self):
        return [self.get_event_title(), self.get_rumour_source_tweets_counts(), self.get_non_rumour_source_tweets_counts(), self.get_rumour_total_tweets_counts(), self.get_non_rumour_total_tweets_counts()]

    def get_event_title(self):
        return self.name.split('/')[-1]

    def get_rumour_total_tweets_counts(self):
        total = self.get_rumour_source_tweets_counts()

        for rumor in self.rumors:
            total += len(rumor.reactions)
        return total

    def get_non_rumour_total_tweets_counts(self):
        total = self.get_non_rumour_source_tweets_counts()

        for non_rumor in self.non_rumors:
            total += len(non_rumor.reactions)
        return total

    def get_rumour_source_tweets_counts(self):
        return len(self.rumors)

    def get_non_rumour_source_tweets_counts(self):
        return len(self.non_rumors)
