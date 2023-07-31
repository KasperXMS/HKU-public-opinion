package com.example.demo.Entity;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class weibo {
    private String mid;

    private String text;

    private int comments;

    private int likes;

    private LocalDateTime time;

    private String location;

    private String keywords;

}
