# The data was scrapped on Oct 20th 2024 at 16:44

library(readr)
subreddits <- read_csv('/Users/sagnikchakravarty/Desktop/UMD_College_Work/Semester 2/SURV622/Assignment/Assignment 2/subreddit_df.csv')
subreddit <- subreddits$subreddit 

library(RedditExtractoR)

keywords <- c('Trump', 
              'Zelensky', 
              'Ukraine', 
              'Russia', 
              'USA', 
              'Cease Fire', 
              'Putin', 
              'Security Gurantees')

reddit_url <- function(x, keyword = keywords) {
  ls <- list()
  n <- length(keyword)
  for (i in 1:n) {
    df <- data.frame(
      find_thread_urls(
        keyword[i], 
        sort_by = 'top',
        subreddit = x,
        period = 'week'
      )
    )
    df$keyword <- keyword[i]  # Add keyword column
    ls[[i]] <- df
  }
  
  Sys.sleep(5)
  return(do.call(rbind, ls))  # Combine all dataframes
}

# Initialize empty list to store dataframes
result_list <- list()

# Loop through subreddits
for(sub in subreddit) {
  # Get thread URLs for current subreddit
  sub_df <- RedditExtractoR::find_thread_urls(
    subreddit = sub,
    sort_by = "top",  # Can change to "new" or "relevance"
    period = "week"   # Time window for results
  )
  
  # Add subreddit identifier column
  sub_df$subreddit <- sub
  
  # Store in list
  result_list[[length(result_list) + 1]] <- sub_df
  
  # Respect API rate limits (2+ seconds between calls)
  Sys.sleep(2)
}

# Combine all dataframes
final_df <- do.call(rbind, result_list)

write.csv(final_df, '/Users/sagnikchakravarty/Desktop/UMD_College_Work/Semester 2/SURV622/Assignment/Assignment 2/threads_df.csv', row.names = FALSE)


url <- final_df$url

for(url in url){
  
}
