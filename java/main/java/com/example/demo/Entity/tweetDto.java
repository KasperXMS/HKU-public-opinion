package com.example.demo.Entity;

import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class tweetDto {
    private String text;



    private int favoriteCount;


    private String score;
}
