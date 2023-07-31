package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.weiboETO;
import com.example.demo.Entity.wordcloud;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface wordclouMapper extends BaseMapper<wordcloud> {
    @Select("${findwordcloud}")
    List<wordcloud> FindWordCloud(@Param("findwordcloud")String sql);
}
