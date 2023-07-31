package com.example.demo.Entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("tweet_hku")
public class tweet {

    @TableId(value = "tweet_id")
    private String tweetId;

    private LocalDateTime time;

    private String text;

    private int retweetCount;

    private int favoriteCount;

    private int quoteCount;

    private String language;

    private String location;

    private String keyword;

    private double positive;

    private double neutral;

    private double negative;

    private int score;
}
