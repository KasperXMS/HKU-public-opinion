package com.example.demo.Entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("frequency")
public class wordcloud {
    private String word;
    private int overview;
}
